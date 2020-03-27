/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "harness.h"
#include "common.h"
#include "pc_types.h"
#include "block.h"
#include "interstate.h"
#include "autotuner.h"

/*int numNodes=0;
void print_treetofile(FILE* fp);
void print_preorder(Node* node, FILE*fp);*/


extern int DIM;
const int K = 1;	// only support K = 1 for now, ignore command line value

Point *points;
Point **ppoints;
Node *root;
int sort_split;
float rad;

int npoints;

#ifdef BLOCK_PROFILE
BlockProfiler profiler;
#endif
#ifdef PARALLELISM_PROFILE
#include "parallelismprofiler.h"
ParallelismProfiler *parallelismProfiler;
#endif

int app_main(int argc, char **argv);
void read_input(int argc, char **argv);
void read_point(FILE *in, Point *p);
void find_correlation(int start, int end);
void correlation_search(Point *point, Node *root);
void correlation_search_block(Node *node, _BlockStack *_stack, int _depth);
void correlation_search_blockAutotune(Node *node, _BlockStack *_stack, int _depth, _Autotuner *_autotuner);
void correlation_search_map_root(Node *node);
bool can_correlate(Point * point, Node * cell);

static inline float distance_axis(Point *a, Point *b, int axis);
static inline float distance(Point *a, Point *b);
Node * construct_tree(Point *points, int start_idx, int end_idx, int depth);
void sort_points(Point *points, int start_idx, int end_idx, int depth);
int compare_point(const void *a, const void *b);

void free_tree(Node *n)
{
	if (n->left != NULL) free_tree(n->left);
	if (n->right != NULL) free_tree(n->right);
	delete n;
}

int app_main(int argc, char **argv) {

	read_input(argc, argv);
	cout << "rad " << rad << endl;

	ppoints = new Point*[npoints];
	for (int i = 0; i < npoints; i++) {
		ppoints[i] = &points[i];
	}

	root = construct_tree(points, 0, npoints - 1, 0);

	/*FILE* fp = fopen("treelog.txt","w+");
	print_treetofile(fp);
	fclose(fp);
	printf("tree details output to treelog.txt.\n");*/

	if(!Harness::get_sort_flag()) {
		// randomize points
		srand(0);
		for (int i = 0; i < npoints; i++) {
			int r = rand() % npoints;
			Point *temp = ppoints[i];
			ppoints[i] = ppoints[r];
			ppoints[r] = temp;
		}
	}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler = new ParallelismProfiler;
#endif
	
	//double startTime, endTime;
	Harness::start_timing();
	//startTime = clock();
	if (Harness::get_block_size() > 0) {
		_Block::max_block = Harness::get_block_size();
		_IntermediaryBlock::max_block = Harness::get_block_size();
	}
	correlation_search_map_root(root);
	Harness::parallel_for(find_correlation, 0, npoints);
	//endTime=clock();
	//double consumedTime = endTime - startTime;
	Harness::stop_timing();

	long long sum = 0;
	for (int i = 0; i < npoints; i++) {
		sum += points[i].corr;
	}
	cout << "avg corr: " << (float)sum / npoints << endl;

#ifdef TRACK_TRAVERSALS
	long int sum_nodes_traversed = 0;
	for (int i = 0; i < npoints; i++) {
		Point *p = &points[i];
		sum_nodes_traversed += p->num_nodes_traversed;
		//printf("%d %d\n", p->id, p->num_nodes_traversed);
	}
	printf("sum_nodes_traversed:%ld\n", sum_nodes_traversed);
	printf("(%f %f %f %f)\n",root->min[0],root->max[0],root->min[1],root->max[1]);
#endif
	//printf("time consumed: %f\n",consumedTime/CLOCKS_PER_SEC);
#ifdef BLOCK_PROFILE
	profiler.output();
#endif
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->output();
	delete parallelismProfiler;
#endif

	delete [] ppoints;
	delete [] points;
	free_tree(root);

	return 0;
}

void find_correlation(int start, int end) {
	if (Harness::get_block_size() == 0) {

		_Autotuner _autotuner(end);
		_BlockStack *_tuneStack = new _BlockStack();
		_Block *_tuneBlock = new _Block();
		_IntermediaryBlock *_tuneInterBlock = new _IntermediaryBlock();
		_tuneInterBlock->block = _tuneBlock;

		int **_tuneIndexes = _autotuner.tune();
		for (int _t = 0; _t < _autotuner.tuneIndexesCnt; _t++) {
			int *_indexes = _tuneIndexes[_t];
			_autotuner.tuneEntryBlock();
			//cout << _autotuner.sampleSizes[_t] << endl;
			_tuneInterBlock->reset();
			for(int _tt = 0; _tt < _autotuner.sampleSizes[_t]; _tt++) {
				int i = _indexes[_tt];
				class Point *p = ppoints[i];
				_tuneBlock ->  add (p);
				struct _IntermediaryState *_interState = _tuneInterBlock->next ();
			}
			_tuneStack ->  get (0) -> block = _tuneBlock;
			//compute_force_recursive_blockAutotune(_Block::root_node,_Block::root_dsq,_Block::root_epssq,_tuneStack,0,&_autotuner);
			correlation_search_blockAutotune(_Block::root_node, _tuneStack, 0, &_autotuner);
			_tuneBlock ->  recycle ();
			_tuneInterBlock->reset();
			for(int _tt = 0; _tt < _autotuner.sampleSizes[_t]; _tt++) {
				struct _IntermediaryState *_interState = _tuneInterBlock->next ();
			}
			_autotuner.tuneExitBlock(_t);
		}
		_autotuner.tuneFinished();
		delete _tuneStack;
		delete _tuneBlock;
		delete _tuneInterBlock;

		auto_ptr<_BlockStack> _stack(new _BlockStack);
		auto_ptr<_Block> _block(new _Block);

		// _inter_block isn't necessary for this benchmark, but it is part of the generated template for point blocking
		auto_ptr<_IntermediaryBlock> _inter_block(new _IntermediaryBlock);
		for (int _start = start; _start < end; _start += _Block::max_block) {
			int _end = min(_start + _Block::max_block, end);
			_inter_block->reset();
			for (int i = _start; i < _end; i++) {
				if(_autotuner.isSampled(i)) continue ;
				_block->add(ppoints[i]);
				//_inter_block->next();
			}
			_stack->get(0)->block = _block.get();
			//cout << _start << endl;
			correlation_search_block(_Block::root_node, _stack.get(), 0);
			_block->recycle();
			_inter_block->reset();
			for (int i = _start; i < _end; i++) {
				if(_autotuner.isSampled(i)) continue ;
				//_inter_block->next();
			}
			//nearest_neighbor_search(&search_points[i], root);
		}
	} else {
		auto_ptr<_BlockStack> _stack(new _BlockStack);
		auto_ptr<_Block> _block(new _Block);

		// _inter_block isn't necessary for this benchmark, but it is part of the generated template for point blocking
		auto_ptr<_IntermediaryBlock> _inter_block(new _IntermediaryBlock);
		for (int _start = start; _start < end; _start += _Block::max_block) {
			int _end = min(_start + _Block::max_block, end);
			_inter_block->reset();
			for (int i = _start; i < _end; i++) {
				_block->add(ppoints[i]);
				//_inter_block->next();
			}
			_stack->get(0)->block = _block.get();
			//cout << _start << endl;
			correlation_search_block(_Block::root_node, _stack.get(), 0);
			_block->recycle();
			_inter_block->reset();
			for (int i = _start; i < _end; i++) {
				//_inter_block->next();
			}
			//nearest_neighbor_search(&search_points[i], root);
		}
	}
}

void correlation_search_map_root(Node *node) {
	_Block::root_node = node;
}

void correlation_search(Point *point, Node *node) {
	assert(node != NULL);

#ifdef TRACK_TRAVERSALS
	point->num_nodes_traversed++;
	//if (point->id == 10000) cout << node->id << " " << point->closest_dist << endl;
#endif

	// is this node closer than the current best?
	if(!can_correlate(point, node)) {
		return;
	}

	if(node->axis == DIM) {
		for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
			Point *candidate = node->points[i];
			if (candidate == NULL) break;
			float dist = distance(point, candidate);
			if (sqrt(dist) < rad) point->corr++;
		}
	} else {
		correlation_search(point, node->left);
		correlation_search(point, node->right);
	}
}

void correlation_search_block(Node *node, _BlockStack *_stack, int _depth) {
	assert(node != NULL);
	_BlockSet *_set = _stack->get(_depth);
	_Block *_block = _set->block;
	_Block *_next_block0 = &_set->nextBlock0;
	_next_block0->recycle();

#ifdef BLOCK_PROFILE
	profiler.record(_block->size);
#endif

	for (int _bi = 0; _bi < _block->size; _bi++) {
		class _Point &_point = _block ->  get (_bi);
		class Point *point = _point._Point::point;

#ifdef TRACK_TRAVERSALS
		point->num_nodes_traversed++;
#endif

		// is this node closer than the current best?
		if(!can_correlate(point, node)) {
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordTruncate();
#endif
			continue;
		}

		if(node->axis == DIM) {
			for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
				Point *candidate = node->points[i];
				if (candidate == NULL) break;
				float dist = distance(point, candidate);
				if (sqrt(dist) < rad) point->corr++;
			}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordTruncate();
#endif
		} else {
			_next_block0->add(point);
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordRecurse();
#endif
		}
	}

	if (!_next_block0->is_empty()) {
		_stack->get(_depth + 1)->block = _next_block0;
		correlation_search_block(node->left, _stack, _depth + 1);
		correlation_search_block(node->right, _stack, _depth + 1);
	}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->blockEnd();
#endif
}

void correlation_search_blockAutotune(Node *node, _BlockStack *_stack, int _depth, _Autotuner *_autotuner) {
	assert(node != NULL);
	_BlockSet *_set = _stack->get(_depth);
	_Block *_block = _set->block;
	_Block *_next_block0 = &_set->nextBlock0;
	_autotuner->profileWorkDone(_block->size);
	_next_block0->recycle();

#ifdef BLOCK_PROFILE
	profiler.record(_block->size);
#endif

	for (int _bi = 0; _bi < _block->size; _bi++) {
		class _Point &_point = _block ->  get (_bi);
		class Point *point = _point._Point::point;

#ifdef TRACK_TRAVERSALS
		point->num_nodes_traversed++;
#endif

		// is this node closer than the current best?
		if(!can_correlate(point, node)) {
			continue;
		}

		if(node->axis == DIM) {
			for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
				Point *candidate = node->points[i];
				if (candidate == NULL) break;
				float dist = distance(point, candidate);
				if (sqrt(dist) < rad) point->corr++;
			}
		} else {
			_next_block0->add(point);
		}
	}

	if (!_next_block0->is_empty()) {
		_stack->get(_depth + 1)->block = _next_block0;
		correlation_search_blockAutotune(node->left, _stack, _depth + 1, _autotuner);
		correlation_search_blockAutotune(node->right, _stack, _depth + 1, _autotuner);
	}
}

bool can_correlate(Point * point, Node * cell) {
	float sum = 0.0;
	float boxsum = 0.0;
	for(int i = 0; i < DIM; i++) {
		float center = (cell->max[i] + cell->min[i]) / 2;
		float boxdist = (cell->max[i] - cell->min[i]) / 2;
		float dist = point->coord[i] - center;
		sum += dist * dist;
		boxsum += boxdist * boxdist;
	}
	if(sqrt(sum) - sqrt(boxsum) < rad)
		return true;
	else
		return false;
}


static inline float distance_axis(Point *a, Point *b, int axis) {
	return (a->coord[axis] - b->coord[axis]) * (a->coord[axis] - b->coord[axis]);
}

static inline float distance(Point *a, Point *b) {
	float d = 0;
	for(int i = 0; i < DIM; i++) {
		d += distance_axis(a,b,i);
	}
	return d;
}

int compare_point(const void *a, const void *b) {
	if(((Point *)a)->coord[sort_split] < ((Point *)b)->coord[sort_split]) {
		return -1;
	} else if(((Point *)a)->coord[sort_split] > ((Point *)b)->coord[sort_split]) {
		return 1;
	} else {
		return 0;
	}
}

float max(float a, float b) {
	return a > b ? a : b;
}

float min(float a, float b) {
	return a < b ? a : b;
}

Node * construct_tree(Point * points, int lb, int ub, int depth) {
	Node *node = new Node;
	/*numNodes++;
	node->id = numNodes;*/
	int size = ub - lb + 1;
	int mid;
	int i, j;

	if (size <= MAX_POINTS_IN_CELL) {
		for (i = 0; i < size; i++) {
			node->points[i] = &points[lb + i];
			for (j = 0; j < DIM; j++) {
				node->max[j] = max(node->max[j], points[lb + i].coord[j]);
				node->min[j] = min(node->min[j], points[lb + i].coord[j]);
				//printf("%f %f %f\n", node->max[j], node->min[j], points[lb + i].coord[j]);

			}
			//exit(0);
		}
		node->axis = DIM; // leaf node has axis of DIM
		return node;

	} else {
		sort_split = depth % DIM;
		qsort(&points[lb], ub - lb + 1, sizeof(Point), compare_point);
		mid = (ub + lb) / 2;

		node->axis = depth % DIM;
		node->splitval = points[mid].coord[node->axis];
		node->left = construct_tree(points, lb, mid, depth + 1);
		node->right = construct_tree(points, mid+1, ub, depth + 1);

		for(j = 0; j < DIM; j++) {
			node->min[j] = min(node->left->min[j], node->right->min[j]);
			node->max[j] = max(node->left->max[j], node->right->max[j]);
			//printf("%f %f %f\n", node->max[j], node->min[j], node->left->min[j]);
		}
		return node;
	}	
}

void sort_points(Point * points, int lb, int ub, int depth) {
	int mid;
	if(lb >= ub)
		return;

	sort_split = depth % DIM;
	qsort(&points[lb], ub - lb + 1, sizeof(Point), compare_point);
	mid = (ub + lb) / 2;

	if(mid > lb) {
		sort_points(points, lb, mid - 1, depth + 1);
		sort_points(points, mid, ub, depth + 1);
	} else {
		sort_points(points, lb, mid, depth + 1);
		sort_points(points, mid+1, ub, depth + 1);
	}
}

void read_input(int argc, char **argv) {
	unsigned long long i, j, k;
	float min = FLT_MAX;
	float max = FLT_MIN;
	FILE *in;

	if(argc != 4 && argc != 3) {
		fprintf(stderr, "usage: nn <DIM> <rad> <npoints> [input_file]\n");
		exit(1);
	}

	DIM = atoi(argv[0]);
	if(DIM <= 0) {
		fprintf(stderr, "Invalid DIM\n");
		exit(1);
	}

	rad = atof(argv[1]);

	npoints = atol(argv[2]);
	if(npoints <= 0) {
		fprintf(stderr, "Not enough points.\n");
		exit(1);
	}


	points = new Point[npoints];

	if(argc == 4) {
		in = fopen(argv[3], "r");
		if(in == NULL) {
			fprintf(stderr, "Could not open %s\n", argv[4]);
			exit(1);
		}

		for(i = 0; i < npoints; i++) {
			read_point(in, &points[i]);
		}
		fclose(in);
	} else {
		srand(0);
		for(i = 0; i < npoints; i++) {
			for(j = 0; j < DIM; j++) {
				points[i].coord[j] = (float)rand() / RAND_MAX;
			}
		}
	}
}

void read_point(FILE *in, Point *p) {
	int dummy;
	/*if(fscanf(in, "%d", &dummy) != 1) {
		fprintf(stderr, "Input file not large enough.\n");
		exit(1);
	}*/
	for(int j = 0; j < DIM; j++) {
		if(fscanf(in, "%f", &p->coord[j]) != 1) {
			fprintf(stderr, "Input file not large enough.\n");
			exit(1);
		}
	}
}

/*void print_treetofile(FILE* fp)
{
	print_preorder(root, fp);
}

void print_preorder(Node* node, FILE* fp)
{
	fprintf(fp,"%d ",node->id);
	for (int j = 0; j < DIM; j++) 
	{
		fprintf(fp,"%f %f ",node->max[j],node->min[j]);
	}
	fprintf(fp,"\n");
	if(node->left)
		print_preorder(node->left,fp);

	if(node->right)
		print_preorder(node->right,fp);
}*/

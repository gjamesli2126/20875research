/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "harness.h"
#include "blocks.h"
#include "interstate.h"
#include "autotuner.h"
//int uniqId;
extern int DIM;
const int K = 1;	// only support K = 1 for now, ignore command line value
/*int numNodes=0;
int totalNodesAtDepth[20];
void print_treetofile(FILE* fp);
void print_preorder(Node* node, FILE*fp);*/

Point *training_points;
Node *root;
Point *search_points;
int sort_split;

int npoints;
int nsearchpoints;

#ifdef BLOCK_PROFILE
BlockProfiler profiler;
#endif
#ifdef PARALLELISM_PROFILE
#include "parallelismprofiler.h"
ParallelismProfiler *parallelismProfiler;
#endif

int app_main(int argc, char **argv);
void read_input(int argc, char **argv);
void read_point(FILE *in, Point *p, int label);
void find_nearest_neighbors(int start, int end);
void nearest_neighbor_search(Point *point, Node *root);
void nearest_neighbor_search_slow(Point *point);
int can_correlate(Point * point, Node * cell, float rad);
void update_closest(Point *point, Point *candidate);


static inline float distance_axis(Point *a, Point *b, int axis);
static inline float distance(Point *a, Point *b);
Node * construct_tree(Point *points, int start_idx, int end_idx, int depth);
void sort_points(Point *points, int start_idx, int end_idx, int depth);
int compare_point(const void *a, const void *b);
void nearest_neighbor_search_map_root(Node *node);
void nearest_neighbor_search_block(Node *node, BlockStack *_stack, int _depth);
void nearest_neighbor_search_blockAutotune(Node *node, BlockStack *_stack, int _depth, _Autotuner *_autotuner);

void free_tree(Node *n)
{
	/*if (n->left != NULL) free_tree(n->left);
	if (n->right != NULL) free_tree(n->right);
	delete n;*/
}

int app_main(int argc, char **argv) {

	read_input(argc, argv);
	printf("configuration: K = %d DIM = %d npoints = %d nsearchpoints = %d\n", K, DIM, npoints, nsearchpoints);


	if(Harness::get_sort_flag()) {
		sort_points(search_points, 0, nsearchpoints - 1, 0);
	}
	
	/*for(int i=0;i<20;i++)
		totalNodesAtDepth[i]=0;*/
	root = construct_tree(training_points, 0, npoints - 1, 0);

	/*FILE* fp = fopen("treelog.txt","w+");
	print_treetofile(fp);
	fclose(fp);
	printf("tree details output to treelog.txt.\n");*/
#ifdef PARALLELISM_PROFILE
	parallelismProfiler = new ParallelismProfiler;
#endif

	double startTime, endTime;
	Harness::start_timing();
	startTime = clock();
	if (Harness::get_block_size() > 0) {
		Block::max_block = Harness::get_block_size();
		_IntermediaryBlock::max_block = Harness::get_block_size();
	}
	nearest_neighbor_search_map_root(root);
	Harness::parallel_for(find_nearest_neighbors, 0, nsearchpoints);
	endTime=clock();
	Harness::stop_timing();
	double consumedTime = endTime - startTime;

	long int correct_cnt = 0;
	for(int i = 0; i < nsearchpoints; i++) {
		if (search_points[i].closest_label == search_points[i].label) {
			correct_cnt++;
		}
	}
	float correct_rate = (float) correct_cnt / nsearchpoints;
	printf("correct rate: %.4f\n", correct_rate);

#ifdef TRACK_TRAVERSALS
	long int sum_nodes_traversed = 0;
	for (int i = 0; i < nsearchpoints; i++) {
		Point *p = &search_points[i];
		sum_nodes_traversed += p->num_nodes_traversed;
		//printf("%d: %d (%f %f %f %f %f %f %f) (%1.3f) \n", i, p->id, p->coord[0],p->coord[1],p->coord[2],p->coord[3],p->coord[4],p->coord[5],p->coord[6],sqrt(p->closest_dist));
		/*if(p->id == 4573)
		{
			for(int j=0;j<p->nodesVisited.size();j++)
				printf("%d\n", p->nodesVisited[j]);
		}*/
	}
	printf("sum_nodes_traversed:%ld\n", sum_nodes_traversed);
	/*printf("numnodes:%d \n", numNodes);
	for(int i=0;i<20;i++)
		printf("Nodes At Depth %d:%d\n",i,totalNodesAtDepth[i]);*/
#endif
	printf("time consumed: %f\n",consumedTime/CLOCKS_PER_SEC);
#ifdef BLOCK_PROFILE
	profiler.output();
#endif
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->output();
	//delete parallelismProfiler;
#endif

	//delete [] training_points;
	//delete [] search_points;
	free_tree(root);

	return 0;
}

void find_nearest_neighbors(int start, int end) {

	if (Harness::get_block_size() == 0) {

		_Autotuner _autotuner(end);
		BlockStack *_tuneStack = new BlockStack();
		Block *_tuneBlock = new Block();
		_IntermediaryBlock *_tuneInterBlock = new _IntermediaryBlock();
		_tuneInterBlock->block = _tuneBlock;

		int **_tuneIndexes = _autotuner.tune();
		for (int _t = 0; _t < _autotuner.tuneIndexesCnt; _t++) {
			int *_indexes = _tuneIndexes[_t];
			_autotuner.tuneEntryBlock();
			_tuneInterBlock->reset();
			for(int _tt = 0; _tt < _autotuner.sampleSizes[_t]; _tt++) {
				int i = _indexes[_tt];
				_tuneBlock->add(&search_points[i]);
				struct _IntermediaryState *_interState = _tuneInterBlock->next ();
			}
			_tuneStack ->  get (0) -> block = _tuneBlock;
			nearest_neighbor_search_blockAutotune(Block::get_root(), _tuneStack, 0, &_autotuner);
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

		auto_ptr<BlockStack> _stack(new BlockStack);
		auto_ptr<Block> _block(new Block);

		for (int _start = start; _start < end; _start += Block::get_max_block()) {
			int _end = min(_start + Block::get_max_block(), end);
			for (int i = _start; i < _end; i++) {
				if(_autotuner.isSampled(i)) continue ;
				_block->add(&search_points[i]);
				//_inter_block->next();
			}
			_stack->get(0)->block = _block.get();
			//cout << _start << endl;
			nearest_neighbor_search_block(Block::get_root(), _stack.get(), 0);
			_block->recycle();
			for (int i = _start; i < _end; i++) {
				if(_autotuner.isSampled(i)) continue ;
				//_inter_block->next();
			}
			//nearest_neighbor_search(&search_points[i], root);
		}

	} else {
		auto_ptr<BlockStack> _stack(new BlockStack);
		auto_ptr<Block> _block(new Block);

		for (int _start = start; _start < end; _start += Block::get_max_block()) {
			int _end = min(_start + Block::get_max_block(), end);
			for (int i = _start; i < _end; i++) {
				_block->add(&search_points[i]);
				//_inter_block->next();
			}
			_stack->get(0)->block = _block.get();
			//cout << _start << endl;
			nearest_neighbor_search_block(Block::get_root(), _stack.get(), 0);
			_block->recycle();
			for (int i = _start; i < _end; i++) {
				//_inter_block->next();
			}
			//nearest_neighbor_search(&search_points[i], root);
		}
	}
}

void nearest_neighbor_search_slow(Point *point) {
	for(int i = 0; i < npoints; i++) {
		update_closest(point, &training_points[i]);
	}
}

void update_closest(Point*point, Point *candidate) {
	float dist = distance(point, candidate);
	if (dist < point->closest_dist) {
		point->closest_label = candidate->label;
		point->closest_dist = dist;
	}
}

void nearest_neighbor_search_map_root(Node *node) {
	Block::map_root(node);
}

void nearest_neighbor_search(Point *point, Node *node) {
	assert(false);
	assert(node != NULL);
#ifdef TRACK_TRAVERSALS
	point->num_nodes_traversed++;
#endif

	// is this node closer than the current best?
	if(!can_correlate(point, node, point->closest_dist)) {
		return;
	}

	if(node->axis == DIM) {
		for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
			Point *candidate = node->points[i];
			if (candidate == NULL) break;
			update_closest(point, candidate);
		}
	} else {
		if(point->coord[node->axis] < (node->splitval)) {
			nearest_neighbor_search(point, node->left);
			nearest_neighbor_search(point, node->right);
		} else {
			nearest_neighbor_search(point, node->right);
			nearest_neighbor_search(point, node->left);
		}
	}
}

void nearest_neighbor_search_block(Node *node, BlockStack *_stack, int _depth) {
	assert(node != NULL);
	BlockSet *_set = _stack->get(_depth);
	Block *_block = _set->block;
	Block *_next_block0 = &_set->next_block[0];
	_next_block0->recycle();
	Block *_next_block1 = &_set->next_block[1];
	_next_block1->recycle();

#ifdef BLOCK_PROFILE
	profiler.record(_block->size);
#endif

	for (int _bi = 0; _bi < _block->size; _bi++) {
		Point *point = _block->points[_bi];
#ifdef TRACK_TRAVERSALS
		point->num_nodes_traversed++;
		//point->nodesVisited.push_back(node->id);
#endif


		// is this node closer than the current best?
		if(!can_correlate(point, node, point->closest_dist)) {
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordTruncate();
#endif
			continue;
		}

		if(node->axis == DIM) {
			for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
				Point *candidate = node->points[i];
				if (candidate != NULL) {
					update_closest(point, candidate);
				}
			}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordTruncate();
#endif
		} else {
			if(point->coord[node->axis] < (node->splitval)) {
				_next_block0->add(point);
			} else {
				_next_block1->add(point);
			}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordRecurse();
#endif
		}
	}

	if (!_next_block0->is_empty()) {
		_stack->get(_depth + 1)->block = _next_block0;
		nearest_neighbor_search_block(node->left, _stack, _depth + 1);
		nearest_neighbor_search_block(node->right, _stack, _depth + 1);
	}
	if (!_next_block1->is_empty()) {
		_stack->get(_depth + 1)->block = _next_block1;
		nearest_neighbor_search_block(node->right, _stack, _depth + 1);
		nearest_neighbor_search_block(node->left, _stack, _depth + 1);
	}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->blockEnd();
#endif
}

void nearest_neighbor_search_blockAutotune(Node *node, BlockStack *_stack, int _depth, _Autotuner *_autotuner) {
	assert(node != NULL);
	BlockSet *_set = _stack->get(_depth);
	Block *_block = _set->block;
	Block *_next_block0 = &_set->next_block[0];
	_next_block0->recycle();
	Block *_next_block1 = &_set->next_block[1];
	_autotuner->profileWorkDone(_block->size);
	_next_block1->recycle();

#ifdef BLOCK_PROFILE
	profiler.record(_block->size);
#endif

	for (int _bi = 0; _bi < _block->size; _bi++) {
		Point *point = _block->points[_bi];
#ifdef TRACK_TRAVERSALS
		point->num_nodes_traversed++;
		/*point->nodesVisited.push_back(node->id);
		if ((point->id == 4573) && (node->id == 15108)) {
			printf("debug break\n");
		}*/
#endif


		// is this node closer than the current best?
		if(!can_correlate(point, node, point->closest_dist)) {
			continue;
		}

		if(node->axis == DIM) {
			for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
				Point *candidate = node->points[i];
				if (candidate != NULL) {
					update_closest(point, candidate);
				}
			}
		} else {
			if(point->coord[node->axis] < (node->splitval)) {
				_next_block0->add(point);
			} else {
				_next_block1->add(point);
			}
		}
	}

	if (!_next_block0->is_empty()) {
		_stack->get(_depth + 1)->block = _next_block0;
		nearest_neighbor_search_blockAutotune(node->left, _stack, _depth + 1, _autotuner);
		nearest_neighbor_search_blockAutotune(node->right, _stack, _depth + 1, _autotuner);
	}
	if (!_next_block1->is_empty()) {
		_stack->get(_depth + 1)->block = _next_block1;
		nearest_neighbor_search_blockAutotune(node->right, _stack, _depth + 1, _autotuner);
		nearest_neighbor_search_blockAutotune(node->left, _stack, _depth + 1, _autotuner);
	}
}

int can_correlate(Point * point, Node * cell, float rad) {
	float dist=0.0;
	float boxdist=0.0;
	float sum=0.0;
	float boxsum=0.0;
	float center=0.0;
	int i;

	for(i = 0; i < DIM; i++) {
		center = (cell->max[i] + cell->min[i]) / 2;
		boxdist = (cell->max[i] - cell->min[i]) / 2;
		dist = point->coord[i] - center;
		sum += dist * dist;
		boxsum += boxdist * boxdist;
	}

	/*if ((point->id == 4573) && (cell->id == 15108)) 
		printf("sqrt(sum) - sqrt(boxsum):%lf sqrt(rad):%lf\n",sqrt(sum) - sqrt(boxsum), sqrt(rad));*/

	if(sqrt(sum) - sqrt(boxsum) < sqrt(rad))
	{
		
		/*if ((point->id == 4573) && (cell->id == 15108)) 
		{
			printf("left:%p,right:%p,axis:%d level:%d splitval:%f\n",cell->left,cell->right,cell->axis,cell->level,cell->splitval);
			printf("CORRELATE:%f boxsum:%f cellmax0:%f cellmin0:%f cellmax1:%f cellmin1:%f rad:%f\n",sum, boxsum, cell->max[0],cell->min[0],cell->max[1],cell->min[1],rad);
		}*/
		return 1;
	}
	else
	{
		/*if ((point->id == 4573) && (cell->id == 15108)) 
		{
			printf("left:%p,right:%p,axis:%d level:%d splitval:%f\n",cell->left,cell->right,cell->axis,cell->level,cell->splitval);
			printf("CANNOT CORRELATE:%f boxsum:%f cellmax0:%f cellmin0:%f cellmax1:%f cellmin1:%f rad:%f\n",sum, boxsum, cell->max[0],cell->min[0],cell->max[1],cell->min[1],rad);
		}*/
		return 0;
	}
}


static inline float distance_axis(Point *a, Point *b, int axis) {
	return (a->coord[axis] - b->coord[axis]) * (a->coord[axis] - b->coord[axis]);
}

static inline float distance(Point *a, Point *b) {
	int i;
	float d = 0;
	for(i = 0; i < DIM; i++) {
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
	node->id = numNodes;
	node->level = depth;
	totalNodesAtDepth[depth]++;*/
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
	float max = -FLT_MAX;
	FILE *in;

	if(argc > 5 || argc < 3) {
		fprintf(stderr, "usage: nn <DIM> <K> <npoints> [<nsearchpoints>] [input_file]\n");
		exit(1);
	}

	DIM = atoi(argv[0]);
	if(DIM <= 0) {
		fprintf(stderr, "Invalid DIM\n");
		exit(1);
	}

	//K = atoi(argv[1]); // only support K = 1 for now
	if(K <= 0) {
		fprintf(stderr, "Invalid K\n");
		exit(1);
	}

	npoints = atol(argv[2]);
	if(npoints <= 0) {
		fprintf(stderr, "Not enough points.\n");
		exit(1);
	}

	if(argc >= 4) {
		nsearchpoints = atoi(argv[3]);
		if(nsearchpoints <= 0) {
			fprintf(stderr, "Not enough search points.");
			exit(1);
		}
	} else {
		nsearchpoints=npoints;
	}


	training_points = new Point[npoints];
	search_points = new Point[nsearchpoints];

	if(argc == 5) {
		in = fopen(argv[4], "r");
		if(in == NULL) {
			fprintf(stderr, "Could not open %s\n", argv[4]);
			exit(1);
		}

		for(i = 0; i < npoints; i++) {
			read_point(in, &training_points[i],i);
		}
		//memcpy(search_points,training_points,sizeof(Point)*npoints); 

		for(i = 0; i < nsearchpoints; i++) {
			read_point(in, &search_points[i],i);
		}
		fclose(in);

	} else {
		srand(0);
		for(i = 0; i < npoints; i++) {
			training_points[i].label = i;
			for(j = 0; j < DIM; j++) {
				training_points[i].coord[j] = (float)rand() / RAND_MAX;
			}
		}

		for(i = 0; i < nsearchpoints; i++) {
			search_points[i].label = npoints + i;
			for(j = 0; j < DIM; j++) {
				search_points[i].coord[j] = (float)rand() / RAND_MAX;
			}
		}
	}
}

void read_point(FILE *in, Point *p, int label) {
	int j;
	/*if(fscanf(in, "%d", &p->label) != 1) {
		fprintf(stderr, "Input file not large enough.\n");
		exit(1);
	}*/
	for(j = 0; j < DIM; j++) {
		if(fscanf(in, "%f", &p->coord[j]) != 1) {
			fprintf(stderr, "Input file not large enough.\n");
			exit(1);
		}
	}
	p->id=label;
	//p->id = uniqId++;
}

/*void print_treetofile(FILE* fp)
{
	print_preorder(root, fp);
}

void print_preorder(Node* node, FILE* fp)
{
	fprintf(fp,"%d %f ",node->id, node->splitval);
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

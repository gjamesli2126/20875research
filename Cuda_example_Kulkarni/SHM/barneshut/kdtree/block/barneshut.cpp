/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "harness.h"
#include "common.h"
#include "bh_types.h"
#include "block.h"
#include "interstate.h"
#include "autotuner.h"

/*int numNodes=0;
void print_treetofile(FILE* fp);
void print_preorder(Node* node, FILE*fp);*/


extern int DIM;

Point *points;
Point **ppoints;
Node *root;
int sort_split;
float rad;

long int npoints;
uint64_t ntimesteps;
float dtime;
float eps;
float tol;
float half_dtime;
float inv_tol_squared;
float eps_squared;
float diameter;
int global_timestep;


#ifdef BLOCK_PROFILE
BlockProfiler profiler;
#endif
#ifdef PARALLELISM_PROFILE
#include "parallelismprofiler.h"
ParallelismProfiler *parallelismProfiler;
#endif

int app_main(int argc, char **argv);
void read_input(int argc, char **argv);
void compute_force(int start, int end);
void recursive_compute_force(Point *point, Node *root);
void recursive_compute_force_block(Node *node, _BlockStack *_stack, int _depth);
void recursive_compute_force_blockAutotune(Node *node, _BlockStack *_stack, int _depth, _Autotuner *_autotuner);
void recursive_compute_force_map_root(Node *node);

Node * construct_tree(Point *points, int start_idx, int end_idx, int depth, float mass, float rad, Vec& cofm);
int compare_point(const void *a, const void *b);
float mydistance(Vec& a, Vec& b) {
	float d = 0;
	
	float  tmp = (a.x - b.x);
	float tmpsq = tmp * tmp;
	d +=tmpsq;
	tmp = (a.y - b.y);
	tmpsq = tmp * tmp;
	d +=tmpsq;
	tmp = (a.z - b.z);
	tmpsq = tmp * tmp;
	d +=tmpsq;
	return sqrt(d);
}

/* based on clrs, chap 7,9.*/
int partition(Point* a, int p, int r, void* comparator)
{
	Point x = a[r];
	int i=p-1;
	for(int j=p;j<=r-1;j++)
	{
		if(compare_point(&(a[j]), &x) == -1)
		{
			i=i+1;
			std::swap(a[i],a[j]);
		}
	}
	std::swap(a[i+1],a[r]);
	return i+1;
}

void find_median(Point* a, int p, int r, int i, void* comparator)
{
	if((p==r)||(p>r))
		return;
	int q = partition(a, p, r, comparator);
	int k = q-p+1;
	if(k==i)
		return; 
	else if(i<k)
		return find_median(a,p,q-1,i, comparator);
	else
		return find_median(a,q+1,r,i-k, comparator);
			
}

void my_nth_element(Point* points, int from, int mid, int to,void* comparator)
{
	find_median(points, from, to, mid, comparator); 
}


inline void update_point(Point *p, Node *root, const Vec& dr, float drsq, float epssq) {
	drsq+=epssq;
	float idr=(1.0/sqrt(drsq));
	float nphi=(root->mass*idr);
	float scale=((nphi*idr)*idr);
	p->acc.x+=(dr.x*scale);
	p->acc.y+=(dr.y*scale);
	p->acc.z+=(dr.z*scale);
}


void compute_cell_params(Point* points, long int lb, long int ub, Vec& cofm, float& mass, float& radius)
{
	float min[DIM],max[DIM];
	for(int i=0;i<DIM;i++)
	{
		min[i]=FLT_MAX;
		max[i]=-FLT_MAX;
	}
	for (long int i = lb; i <= ub; i++) 
	{
		/* compute cofm */
		mass += points[i].mass;
		cofm.x += points[i].mass * points[i].cofm.x;
		cofm.y += points[i].mass * points[i].cofm.y;
		cofm.z += points[i].mass * points[i].cofm.z;
		/* compute bounding box */
		if(min[0] > points[i].cofm.x)
			min[0] = points[i].cofm.x;
		if(min[1] > points[i].cofm.y)
			min[1] = points[i].cofm.y;
		if(min[2] > points[i].cofm.z)
			min[2] = points[i].cofm.z;
		if(max[0] < points[i].cofm.x)
			max[0] = points[i].cofm.x;
		if(max[1] < points[i].cofm.y)
			max[1] = points[i].cofm.y;
		if(max[2] < points[i].cofm.z)
			max[2] = points[i].cofm.z;
	}
	/* compute final cofm */
	cofm.x /= mass;
	cofm.y /= mass;
	cofm.z /= mass;

	/* compute center of the box */
	Vec center(0.);
	for(int i=0;i<DIM;i++)
	{
		float coord_i = (min[i]+max[i])/(float)2;
		if(i==0)
			center.x = coord_i;	
		if(i==1)
			center.y = coord_i;	
		else
			center.z = coord_i;	
	}
	
	/* compute Bmax and Bcel */
	float Bmax=0.,Bcel=0.;
	for (long int i = lb; i <= ub; i++) 
	{
		float bmax = mydistance(cofm, points[i].cofm);
		if(Bmax < bmax)
			Bmax = bmax;
		float bcel = mydistance(center, points[i].cofm);
		if(Bcel < bcel)
			Bcel = bcel;
	}
	/* compute ropen aka radius of cell
	formula based on the PKDGRAV paper: http://hpcc.astro.washington.edu/faculty/trq/brandon/pkdgrav.html */
	radius = Bmax/(sqrt(3) * tol) + Bcel;	

}

void free_tree(Node *n)
{
	if (n->left != NULL) free_tree(n->left);
	if (n->right != NULL) free_tree(n->right);
	delete n;
}

int app_main(int argc, char **argv) {

	read_input(argc, argv);

	ppoints = new Point*[npoints];
	for (long int i = 0; i < npoints; i++) {
		ppoints[i] = &points[i];
	}

	Vec cofm(0.0);
	float rad=0., mass=0.;

	compute_cell_params(points, 0, npoints-1, cofm, mass, rad);
		
	root = construct_tree(points, 0, npoints - 1, 0, mass, rad, cofm);
	/*FILE* fp = fopen("treelog.txt","w+");
	print_treetofile(fp);
	fclose(fp);
	printf("tree details output to treelog.txt.\n");*/

	if(!Harness::get_sort_flag()) {
		// randomize points
		srand(0);
		for (long int i = 0; i < npoints; i++) {
			int r = rand() % npoints;
			Point *temp = ppoints[i];
			ppoints[i] = ppoints[r];
			ppoints[r] = temp;
		}
	}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler = new ParallelismProfiler;
#endif
	
	printf("Time step 0\n");
	Harness::start_timing();
	//startTime = clock();
	if (Harness::get_block_size() > 0) {
		_Block::max_block = Harness::get_block_size();
		_IntermediaryBlock::max_block = Harness::get_block_size();
	}
	recursive_compute_force_map_root(root);
	Harness::parallel_for(compute_force, 0, npoints);
	//endTime=clock();
	//double consumedTime = endTime - startTime;
	Harness::stop_timing();


#ifdef TRACK_TRAVERSALS
	long int sum_nodes_traversed = 0;
	for (long int i = 0; i < npoints; i++) {
		Point *p = &points[i];
		sum_nodes_traversed += p->num_nodes_traversed;
		//printf("%d %d\n", p->id, p->num_nodes_traversed);
		if(p->id == 0)
			printf("position: %f %f %f %f %f %f\n",p->cofm.x,p->cofm.y,p->cofm.z, p->acc.x, p->acc.y, p->acc.z);
	}
	printf("sum_nodes_traversed:%ld\n", sum_nodes_traversed);
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

void compute_force(int start, int end) {
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
			_tuneInterBlock->reset();
			for(int _tt = 0; _tt < _autotuner.sampleSizes[_t]; _tt++) {
				int i = _indexes[_tt];
				class Point *p = ppoints[i];
				_tuneBlock ->  add (p);
				struct _IntermediaryState *_interState = _tuneInterBlock->next ();
			}
			_tuneStack ->  get (0) -> block = _tuneBlock;
			recursive_compute_force_blockAutotune(_Block::root_node, _tuneStack, 0, &_autotuner);
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
			}
			_stack->get(0)->block = _block.get();
			//cout << _start << endl;
			recursive_compute_force_block(_Block::root_node, _stack.get(), 0);
			_block->recycle();
			_inter_block->reset();
			for (int i = _start; i < _end; i++) {
				if(_autotuner.isSampled(i)) continue ;
			}
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
			}
			_stack->get(0)->block = _block.get();
			//cout << _start << endl;
			recursive_compute_force_block(_Block::root_node, _stack.get(), 0);
			_block->recycle();
			_inter_block->reset();
			for (int i = _start; i < _end; i++) {
			}
		}
	}
}

void recursive_compute_force_map_root(Node *node) {
	_Block::root_node = node;
}

void recursive_compute_force(Point *point, Node *node) {
	assert(node != NULL);

#ifdef TRACK_TRAVERSALS
	point->num_nodes_traversed++;
	//if (point->id == 10000) cout << node->id << " " << point->closest_dist << endl;
#endif

	Vec dr;
	dr.x=(node->cofm.x-point->cofm.x);
	dr.y=(node->cofm.y-point->cofm.y);
	dr.z=(node->cofm.z-point->cofm.z);
	float drsq=(((dr.x*dr.x)+(dr.y*dr.y))+(dr.z*dr.z));
	float ropensq = node->ropen * node->ropen;
	
	// is this node closer than the current best?
	if(drsq > ropensq) 
	{
		drsq+=eps_squared;
		update_point(point, node, dr, drsq, eps_squared);
		return;
	}

	if(node->leafNode) {
		for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
			Point *candidate = node->points[i];
			if (candidate == NULL) break;
			if(candidate != point)
				update_point(point, node, dr, drsq, eps_squared);
		}
	} else {
		recursive_compute_force(point, node->left);
		recursive_compute_force(point, node->right);
	}
}

int compare_point(const void *a, const void *b) {
	switch(sort_split)
	{
		case 0:
			if(((Point *)a)->cofm.x < ((Point *)b)->cofm.x) {
				return -1;
			} else if(((Point *)a)->cofm.x > ((Point *)b)->cofm.x) {
				return 1;
			} else {
				return 0;
			}
			break;
		case 1:
			if(((Point *)a)->cofm.y < ((Point *)b)->cofm.y) {
				return -1;
			} else if(((Point *)a)->cofm.y > ((Point *)b)->cofm.y) {
				return 1;
			} else {
				return 0;
			}
			break;
		case 2:
			if(((Point *)a)->cofm.z < ((Point *)b)->cofm.z) {
				return -1;
			} else if(((Point *)a)->cofm.z > ((Point *)b)->cofm.z) {
				return 1;
			} else {
				return 0;
			}
			break;
	}

}


void recursive_compute_force_block(Node *node, _BlockStack *_stack, int _depth) {
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

	Vec dr;
	dr.x=(node->cofm.x-point->cofm.x);
	dr.y=(node->cofm.y-point->cofm.y);
	dr.z=(node->cofm.z-point->cofm.z);
	float drsq=(((dr.x*dr.x)+(dr.y*dr.y))+(dr.z*dr.z));
	float ropensq = node->ropen * node->ropen;
	
	// is this node closer than the current best?
	if(drsq > ropensq) 
	{
		update_point(point, node, dr, drsq, eps_squared);
#ifdef PARALLELISM_PROFILE
		parallelismProfiler->recordTruncate();
#endif
		continue;
	}

	if(node->leafNode) {
		for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
			Point *candidate = node->points[i];
			if (candidate == NULL) break;
			if(candidate != point)
				update_point(point, node, dr, drsq, eps_squared);
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
		recursive_compute_force_block(node->left, _stack, _depth + 1);
		recursive_compute_force_block(node->right, _stack, _depth + 1);
	}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->blockEnd();
#endif
}

void recursive_compute_force_blockAutotune(Node *node, _BlockStack *_stack, int _depth, _Autotuner *_autotuner) {
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

	Vec dr;
	dr.x=(node->cofm.x-point->cofm.x);
	dr.y=(node->cofm.y-point->cofm.y);
	dr.z=(node->cofm.z-point->cofm.z);
	float drsq=(((dr.x*dr.x)+(dr.y*dr.y))+(dr.z*dr.z));
	float ropensq = node->ropen * node->ropen;
	
	// is this node closer than the current best?
	if(drsq > ropensq) 
	{
		drsq+=eps_squared;
		update_point(point, node, dr, drsq, eps_squared);
		continue;
	}

	if(node->leafNode) {
		for (int i = 0; i < MAX_POINTS_IN_CELL; i++) {
			Point *candidate = node->points[i];
			if (candidate == NULL) break;
			if(candidate != point)
				update_point(point, node, dr, drsq, eps_squared);
		}
	} else {
		_next_block0->add(point);
		}
	}

	if (!_next_block0->is_empty()) {
		_stack->get(_depth + 1)->block = _next_block0;
		recursive_compute_force_blockAutotune(node->left, _stack, _depth + 1, _autotuner);
		recursive_compute_force_blockAutotune(node->right, _stack, _depth + 1, _autotuner);
	}
}

Node * construct_tree(Point * points, int lb, int ub, int depth, float mass, float rad, Vec& cofm) {
	Node *node = new Node;
	node->mass = mass;
	(node->cofm).x = cofm.x;
	(node->cofm).y = cofm.y;
	(node->cofm).z = cofm.z;
	node->ropen = rad;

	int size = ub - lb + 1;
	int mid;
	int i, j;

	if (size <= MAX_POINTS_IN_CELL) {
		for (i = 0; i < size; i++) {
			node->points[i] = &points[lb + i];
		}
		node->leafNode = true; // leaf node has axis of DIM
		return node;

	} else {
		/*sort_split = depth % DIM;
		qsort(&points[lb], ub - lb + 1, sizeof(Point), compare_point);*/
		mid = (ub + lb) / 2;
		float min[DIM],max[DIM];
		for(int i=0;i<DIM;i++)
		{
			min[i]=FLT_MAX;
			max[i]=-FLT_MAX;
		}
		for (long int i = lb; i <= ub; i++) 
		{
			/* compute bounding box */
			if(min[0] > points[i].cofm.x)
				min[0] = points[i].cofm.x;
			if(min[1] > points[i].cofm.y)
				min[1] = points[i].cofm.y;
			if(min[2] > points[i].cofm.z)
				min[2] = points[i].cofm.z;
			if(max[0] < points[i].cofm.x)
				max[0] = points[i].cofm.x;
			if(max[1] < points[i].cofm.y)
				max[1] = points[i].cofm.y;
			if(max[2] < points[i].cofm.z)
				max[2] = points[i].cofm.z;
		}
		float maxWidth=max[0]-min[0];
		sort_split=0;
		for(int i=1;i<DIM;i++)
		{
			if((max[i]-min[i]) > maxWidth)
			{
				maxWidth = max[i] - min[i];
				sort_split=i;
			}
		}
		//qsort(&points[lb], ub - lb + 1, sizeof(Point), compare_point);

		int dummy;
		my_nth_element(points, lb, (ub-lb)/2, ub,&dummy);

		float leftMass=0., rightMass=0., leftRad=0., rightRad=0.; 
		Vec leftCofm(0.), rightCofm(0.);
		compute_cell_params(points, lb, mid, leftCofm, leftMass, leftRad);
		compute_cell_params(points, mid+1, ub, rightCofm, rightMass, rightRad);
		
		node->left = construct_tree(points, lb, mid, depth + 1, leftMass, leftRad, leftCofm);
		node->right = construct_tree(points, mid+1, ub, depth + 1, rightMass, rightRad, rightCofm);
		return node;
	}	
}

void read_input(int argc, char **argv) {
	if ((argc!=Arg_num)) {
		fprintf(stderr, "Usage: [input_file] [nbodies]\n");
		exit(1);
	}

	FILE *infile=fopen(argv[Arg_inputfile], "r");
	if (( ! infile)) {
		fprintf(stderr, "Error: could not read input file: %s\n", argv[Arg_inputfile]);
		exit(1);
	}

	npoints=0;
	npoints=atoll(argv[Arg_npoints]);
	if ((npoints<=0))
	{
		fprintf(stderr, "Error: nbodies not valid.\n");
		exit(1);
	}

	printf("Overriding nbodies from input file. nbodies = %lld\n", npoints);
	uint64_t junk;
	fscanf(infile, "%lld", ( & junk));

	if ((npoints<=0))
	{
		fscanf(infile, "%lld", ( & npoints));
		if ((npoints<1))
		{
			fprintf(stderr, "Error: nbodies must be at least 1!\n");
			exit(1);
		}
	}

	fscanf(infile, "%lld", ( & ntimesteps));
	if ((ntimesteps<1))
	{
		fprintf(stderr, "Error: ntimesteps must be at least 1!\n");
		exit(1);
	}

	fscanf(infile, "%f", ( & dtime));
	if ((dtime<=0.0))
	{
		fprintf(stderr, "Error: dtime can not be zero!\n");
		exit(1);
	}

	fscanf(infile, "%f", ( & eps));
	fscanf(infile, "%f", ( & tol));
	half_dtime=(0.5*dtime);
	inv_tol_squared=(1.0/(tol*tol));
	eps_squared=(eps*eps);
	points = new Point[npoints];

	for (long int i=0; i<npoints; i ++ ) {
		int ret = fscanf(infile, "%f %f %f %f %f %f %f", ( & points[i].mass), ( & points[i].cofm.x), ( & points[i].cofm.y), ( & points[i].cofm.z), ( & points[i].vel.x), ( & points[i].vel.y), ( & points[i].vel.z));
		if (ret!=7) {
			fprintf(stderr, "Error: Invalid point (%d).\n", i);
			exit(1);
		}
		points[i].acc.x=(points[i].acc.y=(points[i].acc.z=0.0));
		points[i].id=i;
#ifdef TRACK_TRAVERSALS
		points[i].num_nodes_traversed=0;
#endif
	}
	if ((infile!=stdin)) {
		fclose(infile);
	}
}


/*void print_treetofile(FILE* fp)
{
	print_preorder(root, fp);
}

void print_preorder(Node* node, FILE* fp)
{
	fprintf(fp,"%d ",node->id);
	fprintf(fp,"%f %f %f %f",node->cofm.x,node->cofm.y, node->cofm.z, node->ropen);
	fprintf(fp,"\n");
	if(node->left)
		print_preorder(node->left,fp);

	if(node->right)
		print_preorder(node->right,fp);
}*/

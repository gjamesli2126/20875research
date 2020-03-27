/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <string>
#include <string.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <limits>
#include <vector>
#include "harness.h"
#include "blockprofiler.h"

using namespace std;

#ifdef BLOCK_PROFILE
BlockProfiler profiler;
#endif
#ifdef PARALLELISM_PROFILE
#include "parallelismprofiler.h"
ParallelismProfiler *parallelismProfiler;
#endif

int nsearchpoints;
long int correct_cnt = 0;
//#define TRACK_TRAVERSALS

int DIM;

class Point {
public:
	float *coord;
	int label;
	float tau;
	int closest_label;

	Point() {
		coord = new float[DIM];
#ifdef TRACK_TRAVERSALS
		num_nodes_traversed = 0;
		id = next_id++;
#endif
	}

	~Point() {
		delete coord;
	}

	void print() {
		cout << label << "\t";
		for (int i = 0; i < DIM; i++) {
			cout << coord[i] << "\t";
		}
		cout << endl;
	}
#ifdef TRACK_TRAVERSALS
	int num_nodes_traversed;
	int id;
	static int next_id;
#endif
};

const int track_id = -1;

#ifdef TRACK_TRAVERSALS
int Point::next_id = 0;
#endif

// to avoid conflict with std::distance :/

float mydistance(const class Point *a,const class Point *b)
{
	float d = 0;
	for (int i = 0; i < DIM; i++) {
		float diff = ((a -> Point::coord)[i] - (b -> Point::coord)[i]);
		d += (diff * diff);
	}
	return (sqrt(d));
}
class std::vector< Point * , std::allocator< Point * >  > _items;

struct Node
{
	int index;
	float threshold;
	Node *parent;
	Node* left;
	Node* right;

#ifdef TRACK_TRAVERSALS
	int id;
	static int next_id;
#endif

	Node() :
		index(0), threshold(0.), parent(0), left(0), right(0) {
#ifdef TRACK_TRAVERSALS
		id = next_id++;
#endif
	}

	~Node() {
		delete left;
		delete right;
	}
};

class std::vector< Point * , std::allocator< Point * >  > searchpoints;
#ifdef TRACK_TRAVERSALS
int Node::next_id = 0;
#endif

struct Node *_root;
#include "block.h"
#include "interstate.h"
#include "autotuner.h"
#pragma afterClassDecs

struct DistanceComparator 
{
	const class Point *item;


	inline DistanceComparator(const class Point *item) : item(item)
	{
	}


	inline bool operator()(const class Point *a,const class Point *b)
	{
		return mydistance(((this) -> item),a) < mydistance(((this) -> item),b);
	}
}
;

struct Node *buildFromPoints(int lower,int upper)
{
	if (upper == lower) {
		return 0L;
	}
	struct Node *node = ::new Node ;
	node -> index = lower;
	if ((upper - lower) > 1) {
		// choose an arbitrary point and move it to the start
		//int i = (((int )((((float )(rand())) / 2147483647) * ((upper - lower) - 1))) + lower);
		int i = lower;
		std::swap< Point * > (_items[lower],_items[i]);
		int median = ((upper + lower) / 2);
		// partitian around the median distance
		std::nth_element(
				_items.begin() + lower + 1,
				_items.begin() + median,
				_items.begin() + upper,
				DistanceComparator( _items[lower] ));
		// what was the median?
		node -> Node::threshold = mydistance((_items[lower]),(_items[median]));
		node -> index = lower;
		node->left = buildFromPoints( lower + 1, median );
		if (node->left != NULL) node->left->parent = node;
		node->right = buildFromPoints( median, upper );
		if (node->right != NULL) node->right->parent = node;
	}
	return node;
}

void search_mapRoot(struct Node *node)
{
	_Block::root_node = node;
}

void search_block(struct Node *node,class _BlockStack *_stack,int _depth)
{


	class _BlockSet *_set = _stack ->  get (_depth);
	class _Block *_block = _set -> block;
	class _Block *_nextBlock1 = &_set -> _BlockSet::nextBlock1;
	_nextBlock1 ->  recycle ();
	class _Block *_nextBlock0 = &_set -> _BlockSet::nextBlock0;
	_nextBlock0 ->  recycle ();
#ifdef BLOCK_PROFILE
	profiler.record(_block->size);
#endif
	for (int _bi = 0; _bi < _block -> _Block::size; ++_bi) {
		class _Point &_point = _block ->  get (_bi);
		class Point *target = _point._Point::target;
		if (node == 0L) {
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordTruncate();
#endif
			continue;
		}
#ifdef TRACK_TRAVERSALS
		target->num_nodes_traversed++;
#endif

		if (node->parent != NULL) {
			float upperDist = mydistance( _items[node->parent->index], target );
			if (node->parent->right == node) {
				if (upperDist + target->tau < node->parent->threshold) {
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordTruncate();
#endif
					continue;
				}
			} else if (node->parent->left == node) {
				if (upperDist - target->tau > node->parent->threshold) {
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordTruncate();
#endif
					continue;
				}
			}
		}
		float dist = mydistance((_items[(node -> index)]),target);
		//printf("dist=%g tau=%gn", dist, target->tau );
		if (dist < (target -> Point::tau)) {
			target -> Point::closest_label = ( *_items[(node -> index)]).Point::label;
			target -> Point::tau = dist;
		}
		if (((node -> Node::left) == 0L) && ((node -> Node::right) == 0L)) {
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordTruncate();
#endif
			continue;
		}
		if (dist < (node -> Node::threshold)) {
			_nextBlock0 ->  add (target);
		}
		else {
			_nextBlock1 ->  add (target);
		}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->recordRecurse();
#endif
	}
	if (_nextBlock0 -> _Block::size > 0) {
		_stack ->  get (_depth + 1) -> _BlockSet::block = _nextBlock0;
		{
			search_block((node -> Node::left),_stack,_depth + 1);
			search_block((node -> Node::right),_stack,_depth + 1);
		}
	}
	if (_nextBlock1 -> _Block::size > 0) {
		_stack ->  get (_depth + 1) -> _BlockSet::block = _nextBlock1;
		{
			search_block((node -> Node::right),_stack,_depth + 1);
			search_block((node -> Node::left),_stack,_depth + 1);
		}
	}
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->blockEnd();
#endif
}

void search_blockAutotune(struct Node *node,class _BlockStack *_stack,int _depth, _Autotuner *_autotuner)
{
	class _BlockSet *_set = _stack ->  get (_depth);
	class _Block *_block = _set -> block;
	class _Block *_nextBlock1 = &_set -> _BlockSet::nextBlock1;
	_nextBlock1 ->  recycle ();
	class _Block *_nextBlock0 = &_set -> _BlockSet::nextBlock0;
	_nextBlock0 ->  recycle ();
	_autotuner->profileWorkDone(_block->size);
#ifdef BLOCK_PROFILE
	profiler.record(_block->size);
#endif
	for (int _bi = 0; _bi < _block -> _Block::size; ++_bi) {
		class _Point &_point = _block ->  get (_bi);
		class Point *target = _point._Point::target;
		if (node == 0L)
			continue;
		if (node->parent != NULL) {
			float upperDist = mydistance( _items[node->parent->index], target );
			if (node->parent->right == node) {
				if (upperDist + target->tau < node->parent->threshold) continue;
			} else if (node->parent->left == node) {
				if (upperDist - target->tau > node->parent->threshold) continue;
			}
		}
#ifdef TRACK_TRAVERSALS
		target->num_nodes_traversed++;
#endif
		float dist = mydistance((_items[(node -> index)]),target);
		//printf("dist=%g tau=%gn", dist, target->tau );
		if (dist < (target -> Point::tau)) {
			target -> Point::closest_label = ( *_items[(node -> index)]).Point::label;
			target -> Point::tau = dist;
		}
		if (((node -> Node::left) == 0L) && ((node -> Node::right) == 0L)) {
			continue;
		}
		if (dist < (node -> Node::threshold)) {
			_nextBlock0 ->  add (target);
		}
		else {
			_nextBlock1 ->  add (target);
		}
	}
	if (_nextBlock0 -> _Block::size > 0) {
		_stack ->  get (_depth + 1) -> _BlockSet::block = _nextBlock0;
		{
			search_blockAutotune((node -> Node::left),_stack,_depth + 1, _autotuner);
			search_blockAutotune((node -> Node::right),_stack,_depth + 1, _autotuner);
		}
	}
	if (_nextBlock1 -> _Block::size > 0) {
		_stack ->  get (_depth + 1) -> _BlockSet::block = _nextBlock1;
		{
			search_blockAutotune((node -> Node::right),_stack,_depth + 1, _autotuner);
			search_blockAutotune((node -> Node::left),_stack,_depth + 1, _autotuner);
		}
	}
}

void search( Node* node, Point* target)
{
	if ( node == NULL ) return;
#ifdef TRACK_TRAVERSALS
	target->num_nodes_traversed++;
#endif
	if (node->parent != NULL) {
		float upperDist = mydistance( _items[node->parent->index], target );
		if (node->parent->right == node) {
			if (upperDist + target->tau < node->parent->threshold) return;
		} else if (node->parent->left == node) {
			if (upperDist - target->tau > node->parent->threshold) return;
		}
	}
	float dist = mydistance( _items[node->index], target );
	//printf("dist=%g tau=%gn", dist, target->tau );

	if ( dist < target->tau ) {
		target->closest_label = _items[node->index]->label;
		target->tau = dist;
	}

	if ( node->left == NULL && node->right == NULL ) {
		return;
	}

	if ( dist < node->threshold ) {
		search( node->left, target);
		search( node->right, target);
	} else {
		search( node->right, target);
		search( node->left, target);
	}
}

void create(const class std::vector< Point * , std::allocator< Point * >  > &items)
{
	delete _root;
	_items = items;
	_root = buildFromPoints(0,(items. size ()));
}

void search_entry_mapRoot()
{
	search_mapRoot(_root);
}

void search_entry_prologue(class Point *target,class _IntermediaryBlock *_interBlock)
{
	target -> Point::tau = std::numeric_limits< float > :: max ();
	_interBlock -> _IntermediaryBlock::block ->  add (target);
}

void search_entry(class Point *target)
{
	target -> Point::tau = std::numeric_limits< float > :: max ();
	search( _root, target);
}

void find_nearest_neighbors(int start, int end) {
if (Harness::get_block_size() > 0) {
		_Block::max_block = Harness::get_block_size();
		_IntermediaryBlock::max_block = Harness::get_block_size();
	}
	search_entry_mapRoot();

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
				class Point *target = searchpoints[i];
				search_entry_prologue(target,_tuneInterBlock);
				struct _IntermediaryState *_interState = _tuneInterBlock->next ();
				_interState -> _IntermediaryState::target = target;
			}
			_tuneStack ->  get (0) -> block = _tuneBlock;
			search_blockAutotune(_Block::root_node,_tuneStack,0, &_autotuner);
			_tuneBlock ->  recycle ();
			_tuneInterBlock->reset();
			for(int _tt = 0; _tt < _autotuner.sampleSizes[_t]; _tt++) {
				struct _IntermediaryState *_interState = _tuneInterBlock->next ();
				class Point *target = _interState -> _IntermediaryState::target;
				/*if ((target -> Point::label) == (target -> Point::closest_label)) {
					correct_cnt++;
				}*/
			}
			_autotuner.tuneExitBlock(_t);
		}
		_autotuner.tuneFinished();
		delete _tuneStack;
		delete _tuneBlock;
		delete _tuneInterBlock;

		class _BlockStack *_stack = new _BlockStack;
		class _Block *_block = new _Block;
		class _IntermediaryBlock* _interBlock = new _IntermediaryBlock;
		_interBlock->_IntermediaryBlock::block = _block;
		for (int _start = start; _start < end; _start += _Block::max_block) {
			int _end = min(_start + _Block::max_block,end);
			_interBlock->reset ();
			for (int i = _start; i < _end; ++i) {
				if(_autotuner.isSampled(i)) continue ;
				class Point *target = searchpoints[i];
				search_entry_prologue(target,_interBlock);
				struct _IntermediaryState *_interState = _interBlock->next ();
				_interState -> _IntermediaryState::target = target;
			}
			_stack-> get (0) -> block = _block;
			search_block(_Block::root_node,_stack,0);
			_block->recycle ();
			_interBlock->reset ();
			for (int i = _start; i < _end; ++i) {
				if(_autotuner.isSampled(i)) continue ;
				struct _IntermediaryState *_interState = _interBlock->next ();
				class Point *target = _interState -> _IntermediaryState::target;
				/*if ((target -> Point::label) == (target -> Point::closest_label)) {
					correct_cnt++;
				}*/
			}
		}
		delete _stack;
		delete _block;
		delete _interBlock;
	} else {
		class _BlockStack *_stack = new _BlockStack;
		class _Block *_block = new _Block;
		class _IntermediaryBlock* _interBlock = new _IntermediaryBlock;
		_interBlock->_IntermediaryBlock::block = _block;
		for (int _start = start; _start < end; _start += _Block::max_block) {
			int _end = min(_start + _Block::max_block,end);
			_interBlock->reset ();
			for (int i = _start; i < _end; ++i) {
				class Point *target = searchpoints[i];
				search_entry_prologue(target,_interBlock);
				struct _IntermediaryState *_interState = _interBlock->next ();
				_interState -> _IntermediaryState::target = target;
			}
			_stack-> get (0) -> block = _block;
			search_block(_Block::root_node,_stack,0);
			_block->recycle ();
			_interBlock->reset ();
			for (int i = _start; i < _end; ++i) {
				struct _IntermediaryState *_interState = _interBlock->next ();
				class Point *target = _interState -> _IntermediaryState::target;
				/*if ((target -> Point::label) == (target -> Point::closest_label)) {
					correct_cnt++;
				}*/
			}
		}
		delete _stack;
		delete _block;
		delete _interBlock;
	}
	
}

class Point *read_point(FILE *in, int label)
{
	class Point *p = ::new Point ;
	/*if (fscanf(in,"%d",&p -> Point::label) != 1) {
		exit(1);
	}*/
	for (int j = 0; j < DIM; j++) {
		if (fscanf(in,"%f",((p -> Point::coord) + j)) != 1) {
			exit(1);
		}
	}
	p->id=label;
	return p;
}

class Point *gen_point()
{
	class Point *p = ::new Point ;
	p->label = 0;
	for (int j = 0; j < DIM; j++) {
		(p -> Point::coord)[j] = (((float )(rand())) / 2147483647);
	}
	return p;
}

enum {
	//Arg_binary,
	Arg_dim,
	Arg_K,
	Arg_npoints,
	Arg_nsearchpoints,
	Arg_inputfile,
	Arg_num,
};

void read_input(int argc, char **argv, std::vector<Point*>& points, std::vector<Point*>& searchpoints) {
	if(argc != Arg_num && argc != Arg_num - 1) {
		fprintf(stderr, "usage: vptree <dim> <K> <npoints> <nsearchpoints> [input_file]\n");
		exit(1);
	}
	DIM = atoi(argv[Arg_dim]);
	int npoints = atol(argv[Arg_npoints]);
	if (npoints <= 0) {
		fprintf(stderr, "Not enough points.\n");
		exit(1);
	}

	nsearchpoints = atol(argv[Arg_nsearchpoints]);
	if (argc == Arg_num) {
		char *filename = argv[Arg_inputfile];
		FILE *in = fopen(filename, "r");
		if( in == NULL) {
			fprintf(stderr, "Could not open %s\n", filename);
			exit(1);
		}

		for (int i = 0; i < npoints; i++) {
			points.push_back(read_point(in,i));
		}
		for (int i = 0; i < nsearchpoints; i++) {
			searchpoints.push_back(read_point(in,i));
			//searchpoints.push_back(points[i]);
		}
		fclose(in);
	} else {
		for (int i = 0; i < npoints; i++) {
			points.push_back(gen_point());
		}
		for (int i = 0; i < npoints; i++) {
			searchpoints.push_back(gen_point());
		}
	}
}

struct SortComparator
{
	const int split;
	SortComparator( const int split ) : split(split) {}
	bool operator()(const Point* a, const Point* b) {
		return a->coord[split] < b->coord[split];
	}
};

void sort_points(std::vector<Point*>& points, int lb, int ub, int depth) {
	int size = ub - lb + 1;
	if (size <= 4) {
		return;
	}
	//	cout << lb << " " << ub << " " << depth << endl;
	//	for (int i = 0; i < points.size(); i++) {
	//		points[i]->print();
	//	}
	//	cout << endl;

	/*int split = depth % DIM;
	std::sort (points.begin() + lb, points.begin() + ub + 1, SortComparator(split));
	int mid = (ub + lb) / 2;

	//	for (int i = 0; i < points.size(); i++) {
	//		points[i]->print();
	//	}
	//	cout << endl << endl;

	sort_points(points, lb, mid, depth + 1);
	sort_points(points, mid+1, ub, depth + 1);*/
	std::sort (points.begin() + lb, points.begin() + ub + 1, SortComparator(0));
}

int app_main(int argc,char *argv[])
{
	srand(0);
	class std::vector< Point * , std::allocator< Point * >  > points;
	read_input(argc,argv,points,searchpoints);
	create(points);
	nsearchpoints = (searchpoints. size ());
	if (Harness::get_sort_flag()) {
		//		for (int i = 0; i < nsearchpoints; i++) {
		//			searchpoints[i]->print();
		//		}
		//		cout << endl << endl;
		sort_points(searchpoints, 0, nsearchpoints - 1, 0);
		//		for (int i = 0; i < nsearchpoints; i++) {
		//			searchpoints[i]->print();
		//		}
	}

#ifdef PARALLELISM_PROFILE
	parallelismProfiler = new ParallelismProfiler;
#endif

	Harness::start_timing();
	double startTime = clock();
#if 0 // non blocked path
	for (int i = 0; i < nsearchpoints; i++) {
		Point* target = searchpoints[i];
		search_entry( target);
		if (target->label == target->closest_label) {
			correct_cnt++;
		}
	}
#else
	Harness::parallel_for(find_nearest_neighbors, 0, nsearchpoints);
	double endTime = clock();
	printf("time consumed %lf\n",(endTime-startTime)/CLOCKS_PER_SEC);
#endif
Harness::stop_timing();


#ifdef TRACK_TRAVERSALS
long long sum_nodes_traversed = 0;
for (int i = 0; i < nsearchpoints; i++) {
	Point* p = searchpoints[i];
	sum_nodes_traversed += p->num_nodes_traversed;
	//printf("%d: %d (%f %f %f %f %f %f %f) (%1.3f) \n", i, p->id, p->coord[0],p->coord[1],p->coord[2],p->coord[3],p->coord[4],p->coord[5],p->coord[6],p->tau);

		if (p->label == p->closest_label) {
			correct_cnt++;
		}
	//printf("(%f %f) %d\n", p->coord[0], p->coord[1], p->num_nodes_traversed);
}
printf("sum_nodes_traversed:%lld\n", sum_nodes_traversed);
#endif

float correct_rate = (((float )correct_cnt) / nsearchpoints);
printf("correct rate: %.4f\n",correct_rate);

#ifdef BLOCK_PROFILE
profiler.output();
#endif
#ifdef PARALLELISM_PROFILE
	parallelismProfiler->output();
	delete parallelismProfiler;
#endif

delete _root;
_root = NULL;
return 0;
}

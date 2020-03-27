/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "common.h"
#include "functions.hpp"
#include "BBT_kernel.hpp"
#include "harness.h"
#include "blocks.h"
#include "interstate.h"
#include "autotuner.h"


#ifdef BLOCK_PROFILE
BlockProfiler profiler;
#endif
#ifdef PARALLELISM_PROFILE
#include "parallelismprofiler.h"
ParallelismProfiler *parallelismProfiler;
#endif

int sort_flag = 0;
int check_flag = 0;
int verbose_flag = 0;
int warp_flag = 0;
int ratio_flag = 0;
unsigned int npoints = 0;
unsigned int nsearchpoints = 0;
unsigned int nthreads = 1;

unsigned int K = 1;

datapoint *points = NULL;
datapoint *search_points = NULL;
neighbor *nearest_neighbor = NULL;

node* tree = NULL;
unsigned int max_depth = 0;
unsigned int nnodes = 0;

void find_k_nearest_neighbors(int start, int end);
void nearest_neighbor_search_map_root(Node *node);
void nearest_neighbor_search_block(Node *node, BlockStack *_stack, int _depth);
void nearest_neighbor_search_blockAutotune(Node *node, BlockStack *_stack, int _depth, _Autotuner *_autotuner);
void k_nearest_neighbor_search(node* tree, datapoint* point, int idx);

TIME_INIT(read_data);
TIME_INIT(build_tree);
TIME_INIT(sort);
TIME_INIT(traversal);

int app_main(int argc, char * argv[]) {
    
    TIME_START(read_data);
    read_input(argc, argv);
    TIME_END(read_data);
    datapoint** dataList = new datapoint* [npoints];
    for (int i = 0; i < npoints; i ++) {
        dataList[i] = &points[i];
    }
    TIME_START(build_tree);
    tree = construct_tree(points, 0, npoints - 1, dataList, 0, 1);
    TIME_END(build_tree);
    printf("The max depth is %d, the nodes number is %d, DIM = %d.\n", max_depth, nnodes, DIM);
//    printTree(tree, 0);
    
    TIME_START(sort);
    if (sort_flag) {
        sort_search_points(search_points, 0, nsearchpoints);
    }
//    for (int i = 0; i < nsearchpoints; i ++) {
//        printf("%d: point %d, label %d\n", i, search_points[i].idx, search_points[i].label);
//    }
    TIME_END(sort);

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
    nearest_neighbor_search_map_root(tree);
    TIME_START(traversal);
    Harness::parallel_for(find_k_nearest_neighbors, 0, nsearchpoints);
    endTime=clock();
    Harness::stop_timing();
    double consumedTime = endTime - startTime;
    TIME_END(traversal);
    printf("time consumed: %f\n",consumedTime/CLOCKS_PER_SEC);
    //print_result();
    
    TIME_ELAPSED_PRINT(read_data, stdout);
    TIME_ELAPSED_PRINT(build_tree, stdout);
    TIME_ELAPSED_PRINT(sort, stdout);
    TIME_ELAPSED_PRINT(traversal, stdout);
#ifdef TRACK_TRAVERSALS
	long int sum_nodes_traversed = 0;
	for (int i = 0; i < nsearchpoints; i++) {
		Point *p = &search_points[i];
		sum_nodes_traversed += p->num_nodes_traversed;
	}
    printf("Total nodes traversed %ld\n", sum_nodes_traversed);
#endif
    printf("Traversal time %f seconds\n", traversal_elapsed/1000);

    delete [] points;
    delete [] search_points;
    delete [] dataList;
    
    return 0;
}

void nearest_neighbor_search_map_root(Node *node) {
    Block::map_root(node);
}

void find_k_nearest_neighbors(int start, int end) {

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
        float dist = getDistance(point, node->pivot);
        int pos = point->idx * K;
        neighbor* pair_list = &nearest_neighbor[pos];
        // is this node closer than the current best?
        if(dist > -0.000001 && pair_list->dist <= (dist - node->rad)) {
#ifdef PARALLELISM_PROFILE
    parallelismProfiler->recordTruncate();
#endif
            continue;
        } else if (node->left == NULL && node->right == NULL) {
            if (dist > pair_list->dist)
                return;
            pair_list->dist = dist;
            pair_list->point = node->pivot;
        
            float f_temp = 0.0;
            datapoint* d_temp = NULL;
            for(int n = 1; n < K && (pair_list + n-1)->dist < (pair_list + n)->dist; n++) {
                f_temp = (pair_list + n)->dist;
                d_temp = (pair_list + n)->point;
                (pair_list + n)->dist = (pair_list + n-1)->dist;
                (pair_list + n)->point = (pair_list + n-1)->point;
                (pair_list + n-1)->dist = f_temp;
                (pair_list + n-1)->point = d_temp;
            }
#ifdef PARALLELISM_PROFILE
    parallelismProfiler->recordTruncate();
#endif
        } else {
            float leftPivotDist = getDistance2(point, node->left->pivot);
            float rightPivotDist = getDistance2(point, node->right->pivot);
            if (leftPivotDist < rightPivotDist) {
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

        float dist = getDistance(point, node->pivot);
        int pos = point->idx * K;
        neighbor* pair_list = &nearest_neighbor[pos];
        // is this node closer than the current best?
        if(dist > -0.000001 && pair_list->dist <= (dist - node->rad)) {
            continue;
        } else if (node->left == NULL && node->right == NULL) {
            if (dist > pair_list->dist)
                return;
            pair_list->dist = dist;
            pair_list->point = node->pivot;
        
            float f_temp = 0.0;
            datapoint* d_temp = NULL;
            for(int n = 1; n < K && (pair_list + n-1)->dist < (pair_list + n)->dist; n++) {
                f_temp = (pair_list + n)->dist;
                d_temp = (pair_list + n)->point;
                (pair_list + n)->dist = (pair_list + n-1)->dist;
                (pair_list + n)->point = (pair_list + n-1)->point;
                (pair_list + n-1)->dist = f_temp;
                (pair_list + n-1)->point = d_temp;
            }
        } else {
            float leftPivotDist = getDistance2(point, node->left->pivot);
            float rightPivotDist = getDistance2(point, node->right->pivot);
            if (leftPivotDist < rightPivotDist) {
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

void k_nearest_neighbor_search(node* node, datapoint* point, int pos) {
#ifdef TRACK_TRAVERSALS
    point->num_nodes_traversed++;
#endif
    float dist = getDistance(point, node->pivot);
    neighbor* pair_list = &nearest_neighbor[pos];
    if (dist > -0.000001 && pair_list->dist <= (dist - node->rad))
        return;
    else if (node->left == NULL && node->right == NULL) {
        if (dist > pair_list->dist)
            return;
        pair_list->dist = dist;
        pair_list->point = node->pivot;
    
        float f_temp = 0.0;
        datapoint* d_temp = NULL;
        for(int n = 1; n < K && (pair_list + n-1)->dist < (pair_list + n)->dist; n++) {
            f_temp = (pair_list + n)->dist;
            d_temp = (pair_list + n)->point;
            (pair_list + n)->dist = (pair_list + n-1)->dist;
            (pair_list + n)->point = (pair_list + n-1)->point;
            (pair_list + n-1)->dist = f_temp;
            (pair_list + n-1)->point = d_temp;
        }
    } else {
        float leftPivotDist = getDistance2(point, node->left->pivot);
        float rightPivotDist = getDistance2(point, node->right->pivot);
        if (leftPivotDist < rightPivotDist) {
            k_nearest_neighbor_search(node->left, point, pos);
            k_nearest_neighbor_search(node->right, point, pos);
        } else {
            k_nearest_neighbor_search(node->right, point, pos);
            k_nearest_neighbor_search(node->left, point, pos);
        }
    }
}

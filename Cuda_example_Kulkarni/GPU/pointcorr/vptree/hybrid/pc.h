#ifndef __VPTREE_H_
#define __VPTREE_H_

#ifndef NUM_OF_WARPS_PER_BLOCK
#define NUM_OF_WARPS_PER_BLOCK 			6
#endif 

#ifndef NUM_OF_BLOCKS
#define NUM_OF_BLOCKS 					(1024)
#endif 

#define NUM_OF_THREADS_PER_WARP 		32
#define NUM_OF_THREADS_PER_BLOCK 		NUM_OF_WARPS_PER_BLOCK * NUM_OF_THREADS_PER_WARP

#define WARP_INDEX (threadIdx.x >> 5)
#ifndef DIM
#define DIM 7
#endif

typedef struct _targs {
	struct Point **searchpoints;
	int lb;
	int ub;
	int tid;
} targs;

struct Point {
	float coord[DIM]; 
	int label;
	float tau;
	int closest_label;
	float vantage_dist; // for building the tree
	long corr;

	#ifdef TRACK_TRAVERSALS
	int num_nodes_traversed;
	int num_trunc;
	int id;
	#endif
};

struct Node {
	struct Point *point;
	float threshold;
	struct Node *parent;
	struct Node *left;
	struct Node *right;
	int pre_id;
	int depth;

	#ifdef TRACK_TRAVERSALS
	int id;
	#endif
};

struct __GPU_Node {
  int point;
  float threshold;
  int parent;
  int left;
  int right;
  int id;
  int pre_id;
  int depth;
};

struct __GPU_point {
  int target;
};

struct __GPU_stack_item {
  int node_index;
};

struct __GPU_tree {
  struct __GPU_Node *nodes;
  struct __GPU_stack_item *stack;
  int nnodes;
  int depth;
  int npoints;
};

float mydistance(struct Point* a, struct Point* b);
struct Node* buildFromPoints(struct Point **_items, int lower, int upper );
void search( struct Node* node, struct Point* target);
void create(int num);
void read_input(int argc, char **argv, struct Point*** points, struct Point*** searchpoints);
int compare_point(const void* p1, const void* p2);
void print_tree(struct Node *root);

void __GPU_findTreeDepth(struct ptrtab *ptab,struct Node *cpu_root,int *nnodes,int *depth,int cur_depth);
int __GPU_copyTree(struct ptrtab *ptab,struct Node *cpu_root,struct __GPU_Node *gpu_root,int *index);
struct __GPU_tree __GPU_buildTree(struct Node *cpu_root,int npoints);
struct __GPU_tree __GPU_allocDeviceTree(struct __GPU_tree gpu_tree_h);
void __GPU_freeDeviceTree(struct __GPU_tree gpu_tree_d);
void __GPU_memcpyTreeToDev(struct __GPU_tree gpu_tree_h,struct __GPU_tree gpu_tree_d);
void __GPU_memcpyTreeToHost(struct __GPU_tree gpu_tree_h,struct __GPU_tree gpu_tree_d);

__global__ void init_kernel(void);
__global__ void search_pre_kernel(struct __GPU_tree d_tree, struct Point *__GPU_point_Point_d, struct __GPU_point *__GPU_point_array_d,struct Point *__GPU_node_point_d, int* d_matrix, int start, int end, int interval, float radius);
__global__ void search_kernel(struct __GPU_tree d_tree, struct Point *__GPU_point_Point_d, struct __GPU_point *__GPU_point_array_d,struct Point *__GPU_node_point_d, int* index_buffer, float radius);

#endif

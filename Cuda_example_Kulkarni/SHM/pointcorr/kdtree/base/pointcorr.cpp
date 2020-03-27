/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "harness.h"
#include "common.h"
#include "pc_types.h"


#ifdef METRICS
std::vector<Node*> subtrees;
class subtreeStats
{
public:
	long int footprint;
	int numnodes;
	subtreeStats(){footprint=0;numnodes=0;}
};
int splice_depth=10;
void getSubtreeStats(Node* ver, subtreeStats* stat);
void printLoadDistribution();
#endif

extern int DIM;
const int K = 1;	// only support K = 1 for now, ignore command line value

Point *points;
Point **ppoints;
Node *root;
int sort_split;
float rad;

int npoints;

int app_main(int argc, char **argv);
void read_input(int argc, char **argv);
void read_point(FILE *in, Point *p);
void find_correlation(int start, int end);
void correlation_search(Point *point, Node *root);
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

	Harness::start_timing();
	Harness::parallel_for(find_correlation, 0, npoints);
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
#endif
#ifdef METRICS
	printLoadDistribution();
#endif


	delete [] ppoints;
	delete [] points;
	free_tree(root);

	return 0;
}

void find_correlation(int start, int end) {
	for(int i = start; i < end; i++) {
		//cout << i << endl;
		correlation_search(ppoints[i], root);
	}
}

void correlation_search(Point *point, Node *node) {
	assert(node != NULL);

#ifdef TRACK_TRAVERSALS
	point->num_nodes_traversed++;
	//if (point->id == 10000) cout << node->id << " " << point->closest_dist << endl;
#endif
	
#ifdef METRICS
	node->numPointsVisited++;
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
#ifdef METRICS
		if(depth == splice_depth-1)
			subtrees.push_back(node);
#endif


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


#ifdef METRICS
void printLoadDistribution()
{
	printf("num bottom subtrees %d\n",subtrees.size());
	std::vector<Node*>::iterator iter = subtrees.begin();
	for(;iter != subtrees.end();iter++)
	{
		long int num_vertices=0, footprint=0;
		subtreeStats stats;
		getSubtreeStats(*iter, &stats);
		printf("(%p) num_vertices %d footprint %ld\n",*iter, stats.numnodes, stats.footprint);
	}
}

void getSubtreeStats(Node* ver, subtreeStats* stats)
{
		stats->numnodes += 1;
		stats->footprint += ver->numPointsVisited;
		assert(ver != NULL);

		if((ver->left == NULL) && (ver->right==NULL))
		{
			return;
		}

		if(ver->left)
		{
			getSubtreeStats(ver->left,stats);
		}
		if(ver->right)
		{
			getSubtreeStats(ver->right,stats);
		}
}


#endif

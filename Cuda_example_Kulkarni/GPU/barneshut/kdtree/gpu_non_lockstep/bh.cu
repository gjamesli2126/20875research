/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "harness.h"

#include "util_common.h"
#include "float.h"
#include "bh.h"
#include "bh_block.h"
#include "bh_kernel_mem.h"
#include "bh_kernel.h"
#include "bh_gpu_tree.h"
#include <unistd.h>

#define DIM 3

Point *points;
Point **ppoints;
Node *root;
int sort_split;
float rad;

long int npoints;
unsigned int ntimesteps;
float dtime;
float eps;
float tol;
float half_dtime;
float inv_tol_squared;
float eps_squared;
float diameter;
int global_timestep;

int sort_flag = 0, verbose_flag = 0, check_flag = 0;
int ratio_flag = 0;
int warp_flag = 0;

int numNodes=0;
bh_gpu_tree* gpu_root_h;
bh_gpu_tree* gpu_root_d;
Point *points_d;

TIME_INIT(read_input);
TIME_INIT(construct_tree);
TIME_INIT(traversal);
TIME_INIT(kernel);

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

int partition(Point* a, int p, int r, void* comparator)
{
	Point x = a[r];
	int i=p-1;
	for(int j=p;j<=r-1;j++)
	{
		if(compare_point(&(a[j]), &x) == -1)
		{
			i=i+1;
			Point ptemp = a[i];
			a[i] = a[j];
			a[j] = ptemp;
		}
	}
	i = i + 1;
	Point ptemp = a[i];
	a[i] = a[r];
	a[r] = ptemp;
	return i;
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
	Vec center;
	center.x = center.y = center.z = 0.0f;
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

int main(int argc, char **argv) {
	TIME_START(read_input);
	read_input(argc, argv);
	TIME_END(read_input);

	ppoints = (Point**) malloc(sizeof(Point*) * npoints);
	for (long int i = 0; i < npoints; i++) {
		ppoints[i] = &points[i];
	}

	Vec cofm;
	cofm.x = cofm.y = cofm.z = 0.0f;
	float rad=0., mass=0.;

	compute_cell_params(points, 0, npoints-1, cofm, mass, rad);
		
	TIME_START(construct_tree);
	root = construct_tree(points, 0, npoints - 1, 0, mass, rad, cofm);
	TIME_END(construct_tree);
	
	/*FILE* fp = fopen("treelog.txt","w+");
	print_treetofile(fp);
	fclose(fp);
	printf("tree details output to treelog.txt.\n");*/

	if(!sort_flag) {
		// randomize points
		srand(0);
		for (long int i = 0; i < npoints; i++) {
			int r = rand() % npoints;
			Point *temp = ppoints[i];
			ppoints[i] = ppoints[r];
			ppoints[r] = temp;
		}
		
		Point* points_h = (Point*)malloc(sizeof(Point) * npoints);
		for (int  i = 0; i < npoints; i ++) {
			points_h[i] = *ppoints[i];
		}
		free(points);
		points = points_h;
	}

	// 1. build gpu tree
	TIME_START(traversal);
	gpu_root_h = build_gpu_tree(root);
	printf("GPU Tree nodes = %d, depth = %d\n", gpu_root_h->nnodes, gpu_root_h->depth);
	printf("gpu_root_h->nodes0[1].cofm.x = %f\n", gpu_root_h->nodes0[1].cofm.x);
	printf("gpu_root_h->nodes0[1].cofm.y = %f\n", gpu_root_h->nodes0[1].cofm.y);
	printf("gpu_root_h->nodes0[1].cofm.z = %f\n", gpu_root_h->nodes0[1].cofm.z);
	// 2. copy tree to GPU
	allocate_gpu_tree_device(gpu_root_h, &gpu_root_d);
	copy_gpu_tree_to_device(gpu_root_h, gpu_root_d);
	// 3. copy points to GPU
	CUDA_SAFE_CALL(cudaMalloc(&points_d, sizeof(Point)*npoints));
	CUDA_SAFE_CALL(cudaMemcpy(points_d, points, sizeof(Point)*npoints, cudaMemcpyHostToDevice));
	// 4. start computation
	init_kernel<<<1,1>>>();
	cudaThreadSynchronize();		
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess) {
		fprintf(stderr, "Error: init_kernel failed with error: %s\n", cudaGetErrorString(e));
		exit(1);
	}
	
	dim3 blocks(NUM_THREAD_BLOCKS);
	dim3 tpb(THREADS_PER_BLOCK);
	// bh_gpu_tree* root, Point* points, int npoints, float eps_squared, float idr, float nphi, int step
	TIME_START(kernel);
	compute_force_gpu<<<blocks, tpb>>>(*gpu_root_d, points_d, npoints, eps_squared, half_dtime, 0);
   	cudaThreadSynchronize();	 
	e = cudaGetLastError();
	if(e != cudaSuccess) {
		fprintf(stderr, "Error: compute_force_gpu failed with error: %s\n", cudaGetErrorString(e));
		exit(1);
	}
	TIME_END(kernel);
	// 5. copy points back to CPU
	CUDA_SAFE_CALL(cudaMemcpy(points, points_d, sizeof(Point)*npoints, cudaMemcpyDeviceToHost));
	TIME_END(traversal);

#ifdef TRACK_TRAVERSALS
	long int sum_nodes_traversed = 0;
	for (long int i = 0; i < npoints; i++) {
		Point *p = &points[i];
		sum_nodes_traversed += p->num_nodes_traversed;
		//printf("%d %d\n", p->id, p->num_nodes_traversed);
	}
	printf("sum_nodes_traversed:%ld, avg num of nodes traversed = %f\n", sum_nodes_traversed, (float)sum_nodes_traversed/npoints);


	if (warp_flag) {
		int maximum = 0, all = 0, j = 0;
		unsigned long long maximum_sum = 0, all_sum = 0;
		int num_of_warps = 0;
        for(int i = 0; i < npoints + (npoints % 32); i+=32) {	

			maximum = points[i].num_nodes_traversed;
			all = points[i].num_nodes_traversed;
			for(j = i + 1; j < i + 32 && j < npoints; j++) {
				if(points[j].num_nodes_traversed > maximum)
					maximum = points[j].num_nodes_traversed;
				all += points[j].num_nodes_traversed;
			}
//			printf("%d\n", maximum);
			maximum_sum += maximum;
			all_sum += all;
            num_of_warps ++;
		}
        printf("avg num of traversed nodes per warp is %f\n", (float)maximum_sum / num_of_warps);
	}
#endif

	if (verbose_flag) {
		for (long int i = 0; i < npoints; i++) {
			Point *p = &points[i];
			printf("i = %d, id = %d, acc.x = %f, acc.y = %f, acc.z = %f\n", i, p->id, p->acc.x, p->acc.y, p->acc.z);
		}
	}

	TIME_ELAPSED_PRINT(read_input, stdout);
	TIME_ELAPSED_PRINT(construct_tree, stdout);
	TIME_ELAPSED_PRINT(traversal, stdout);
	TIME_ELAPSED_PRINT(kernel, stdout);

	delete [] ppoints;
	delete [] points;
	free_tree(root);

	return 0;
}


Node * construct_tree(Point * points, int lb, int ub, int depth, float mass, float rad, Vec& cofm) {
	Node *node = (Node*) malloc (sizeof(Node));
	node->left = node->right = NULL;
	node->leafNode = false;
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
			node->point_id = points[lb + i].id;
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
		Vec leftCofm, rightCofm;
		leftCofm.x = leftCofm.y = leftCofm.z = 0.0f;
		rightCofm.x = rightCofm.y = rightCofm.z = 0.0f;
		compute_cell_params(points, lb, mid, leftCofm, leftMass, leftRad);
		compute_cell_params(points, mid+1, ub, rightCofm, rightMass, rightRad);
		
		node->left = construct_tree(points, lb, mid, depth + 1, leftMass, leftRad, leftCofm);
		node->right = construct_tree(points, mid+1, ub, depth + 1, rightMass, rightRad, rightCofm);
		return node;
	}	
}

void read_input(int argc, char **argv) {
	int i, junk, c;
	if(argc < 1 || argc > 7) {
		fprintf(stderr, "Usage: bh [-c] [-v] [-s] [input_file] [nbodies]\n");
		exit(1);
	}

	check_flag = 0;
	sort_flag = 0;
	verbose_flag = 0;	
	while((c = getopt(argc, argv, "cvsrw")) != -1) {
		switch(c) {
		case 'c':
			check_flag = 1;
			break;
		case 'v':
			verbose_flag = 1;
			break;
		case 's':
			sort_flag = 1;
			break;
        case 'r':
            ratio_flag = 1;
            break;
        case 'w':
            warp_flag = 1;
            break;
		case '?':
			fprintf(stderr, "Error: unknown option.\n");
			exit(1);
			break;
		default:
			abort();
		}
	}

	FILE * infile = stdin;
	npoints = 0;
	for(i = optind; i < argc; i++) {
		switch(i - optind) {
		case 0:
			infile = fopen(argv[i], "r");
			if(!infile) {
				fprintf(stderr, "Error: could not read input file: %s\n", argv[i]);
				exit(1);
			}
			break;

		case 1:
			npoints = atoll(argv[i]);
			if(npoints <= 0) {
				fprintf(stderr, "Error: npoints not valid.\n");
				exit(1);
			}
			printf("Overriding npoints from input file. npoints = %d\n", npoints);
			fscanf(infile, "%d", &junk); // chomp the input size so
			break;
		}
	}

	if ((npoints<=0)) {
		fscanf(infile, "%lld", ( & npoints));
		if ((npoints<1))
		{
			fprintf(stderr, "Error: nbodies must be at least 1!\n");
			exit(1);
		}
	}

	fscanf(infile, "%lld", ( & ntimesteps));
	if ((ntimesteps<1)) {
		fprintf(stderr, "Error: ntimesteps must be at least 1!\n");
		exit(1);
	}

	fscanf(infile, "%f", ( & dtime));
	if ((dtime<=0.0)) {
		fprintf(stderr, "Error: dtime can not be zero!\n");
		exit(1);
	}

	fscanf(infile, "%f", ( & eps));
	fscanf(infile, "%f", ( & tol));
	half_dtime=(0.5*dtime);
	inv_tol_squared=(1.0/(tol*tol));
	eps_squared=(eps*eps);
	points = (Point*) malloc(sizeof(Point) * npoints);

	for (long int i=0; i<npoints; i ++ ) {
		int ret = fscanf(infile, "%f %f %f %f %f %f %f", ( & points[i].mass), ( & points[i].cofm.x), ( & points[i].cofm.y), ( & points[i].cofm.z), ( & points[i].vel.x), ( & points[i].vel.y), ( & points[i].vel.z));
		if (ret!=7) {
			fprintf(stderr, "Error: Invalid point (%d).\n", i);
			exit(1);
		}
		points[i].acc.x=(points[i].acc.y=(points[i].acc.z=0.0));
		points[i].id = i;
#ifdef TRACK_TRAVERSALS
		points[i].num_nodes_traversed=0;
#endif
	}
	if ((infile!=stdin)) {
		fclose(infile);
	}
}


void print_treetofile(FILE* fp)
{
	print_preorder(root, fp);
}

void print_preorder(Node* node, FILE* fp)
{
#ifdef TRACK_TRAVERSALS
	fprintf(fp,"%d ",node->id);
#endif
	fprintf(fp,"%f %f %f %f",node->cofm.x,node->cofm.y, node->cofm.z, node->ropen);
	fprintf(fp,"\n");
	if(node->left)
		print_preorder(node->left,fp);

	if(node->right)
		print_preorder(node->right,fp);
}


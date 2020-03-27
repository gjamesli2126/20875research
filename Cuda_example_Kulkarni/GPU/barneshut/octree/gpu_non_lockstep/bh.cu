/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

#include "util_common.h"

#include "bh.h"
#include "bh_block.h"
#include "bh_kernel_mem.h"
#include "bh_kernel.h"
#include "ptrtab.h"
#include "bh_kernel_params.h"
#include "bh_gpu_tree.h"

unsigned int nbodies; // number of bodies (points)
unsigned int ntimesteps; // number of simulation time steps
float dtime; // delta t
float eps; // softening parameter
float tol; // tolerance <= 0.57
float half_dtime; // dtime / 2
float inv_tol_squared; // 1.0 / tol^2
float eps_squared;

vec3d center; // center of the universe
float diameter; // diameter of the universe box

bh_oct_tree_node* points; // array of simulation points
bh_oct_tree_node** points_sorted; // array of simulation points sorted spatially
bh_oct_tree_node** points_unsorted;
bh_oct_tree_node** points_array;

unsigned int sortidx;

bh_oct_tree_node *root; // root of the OctTree that represents our universe
bh_oct_tree_node **cpu_nodes; // array of cpu nodes for the cpu tree
bh_oct_tree_gpu *h_gpu_root; // root of the GPU OctTree
bh_oct_tree_gpu *d_gpu_root; // root of the GPU OctTree that resides on the GPU
bh_kernel_stacks *d_gpu_stacks; // stack space of the GPU that resides on the GPU

int sort_flag, verbose_flag, check_flag;
int ratio_flag = 0;
int warp_flag = 0;

TIME_INIT(overall);
TIME_INIT(read_input);
TIME_INIT(time_step);
TIME_INIT(compute_bb);
TIME_INIT(build_tree);
TIME_INIT(build_tree_gpu);
TIME_INIT(compute_cofm);
TIME_INIT(force_calc);
TIME_INIT(adv_point);
TIME_INIT(init_kernel);
TIME_INIT(gpu_copy_tree_to);
TIME_INIT(pointer_table);
TIME_INIT(kernel);
TIME_INIT(gpu_copy_tree_from);

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
	nbodies = 0;
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
			nbodies = atoll(argv[i]);
			if(nbodies <= 0) {
				fprintf(stderr, "Error: nbodies not valid.\n");
				exit(1);
			}
			printf("Overriding nbodies from input file. nbodies = %d\n", nbodies);
			fscanf(infile, "%d", &junk); // chomp the input size so
			break;
		}
	}

	if(nbodies <= 0) {
		fscanf(infile, "%d", &nbodies);
		if (nbodies < 1) {
			fprintf(stderr, "Error: nbodies must be at least 1!\n");
			exit(1);
		}
	}

	fscanf(infile, "%d", &ntimesteps);
	if (ntimesteps < 1) {
		fprintf(stderr, "Error: ntimesteps must be at least 1!\n");
		exit(1);
	}

	fscanf(infile, "%f", &dtime);
	if (dtime <= 0.0) {
		fprintf(stderr, "Error: dtime can not be zero!\n");
		exit(1);
	}

	fscanf(infile, "%f", &eps);
	fscanf(infile, "%f", &tol);

	half_dtime = 0.5 * dtime;
	inv_tol_squared = 1.0 / (tol * tol);
	eps_squared = eps * eps;

	/* Read the points */
	points = (bh_oct_tree_node*) calloc(nbodies, sizeof(bh_oct_tree_node));
	if (points == 0) {
		fprintf(stderr, "Error: Could not allocate space for bodies!\n");
		exit(1);
	}

	points_sorted = (bh_oct_tree_node**) calloc(nbodies, sizeof(bh_oct_tree_node*));
	points_unsorted = (bh_oct_tree_node**) calloc(nbodies, sizeof(bh_oct_tree_node*));
	sortidx = 0; // reset sort index
	if (points_sorted == 0) {
		fprintf(stderr, "Error: Could not allocate space for sorting bodies!\n");
		exit(1);
	}

	for (i = 0; i < nbodies; i++) {
		points[i].type = bhLeafNode;

		if (fscanf(infile, "%f %f %f %f %f %f %f", &(points[i].mass),
				&(points[i].cofm.x), &(points[i].cofm.y), &(points[i].cofm.z),
				&(points[i].vel.x), &(points[i].vel.y), &(points[i].vel.z)) != 7) {
			fprintf(stderr, "Error: Invalid point (%d).\n", i);
			exit(1);
		}

		points[i].acc.x = points[i].acc.y = points[i].acc.z = 0.0; // no initial acceleration
		clear_children(&(points[i])); // leaf nodes have no children

		points[i].id = i; // mainly to debug and keep track of points
		points_unsorted[i] = &points[i];
	}

	if (infile != stdin) {
		fclose(infile);
	}
}

void compute_bounding_box(void) {
	vec3d min, max, pos;
	int i;

	min.x = min.y = min.z = FLT_MAX;
	max.x = max.y = max.z = FLT_MIN;

	/* compute the max and min positions to form a bounding box */
	for (i = 0; i < nbodies; i++) {
		pos=points[i].cofm;

		if (min.x > pos.x)
			min.x = pos.x;

		if (min.y > pos.y)
			min.y = pos.y;

		if (min.z > pos.z)
			min.z = pos.z;

		if (max.x < pos.x)
			max.x = pos.x;

		if (max.y < pos.y)
			max.y = pos.y;

		if (max.z < pos.z)
			max.z = pos.z;
	}

	/* compute the maximum of the diameters of all axes */
	diameter = max.x - min.x;

	if (diameter < (max.y - min.y))
		diameter = max.y - min.y;

	if (diameter < (max.z - min.z))
		diameter = max.z - min.z;

	/* compute the center point */
	center.x = max.x + min.x;
	center.y = max.y + min.y;
	center.z = max.z + min.z;

	center.x = center.x * 0.5;
	center.y = center.y * 0.5;
	center.z = center.z * 0.5;
}

void insert_point(bh_oct_tree_node *root, bh_oct_tree_node *p, float r) {
	vec3d offset;
	offset.x = offset.y = offset.z = 0.0;

	assert(root != 0);
	assert(p != 0);

	/*
	 * From the root locate where this point will be:
	 * 		- 0 is lower-front-left
	 * 		- 1 is lower-front-right
	 * 		- 2 is lower-back-left
	 * 		- 3 is lower-back-right
	 * 		- 4 is upper-front-left
	 * 		- 5 is upper-front-right
	 * 		- 6 is upper-back-left
	 * 		- 7 is upper-back-right
	 *
	 * 1. Starting from the lower, front left compare the x component:
	 * 		- if p.x is greater than root.x then go right (add 1)
	 * 		- else stay left
	 *
	 * 		- if p.y is greater than root.y then go back (add 2)
	 * 		- else stay front
	 *
	 * 		- if p.z is greater than root.z then go up (add 4)
	 * 		- else stay low
	 */

	int space = 0; // to place point
	if (root->cofm.x < p->cofm.x) {
		space = 1;
		offset.x = r;
	}

	if (root->cofm.y < p->cofm.y) {
		space += 2;
		offset.y = r;
	}

	if (root->cofm.z < p->cofm.z) {
		space += 4;
		offset.z = r;
	}

	bh_oct_tree_node * child = root->children[space];

	// There was no child here
	if (child == 0) {
		root->children[space] = p;
	} else {
		float half_r = 0.5 * r;

		if (child->type == bhLeafNode) {

			// If we reach a leaf node then we must further divide the space
			// Note there is no need to allocate a new node, we will just reuse the
			// current leaf
			bh_oct_tree_node * new_inner_node = (bh_oct_tree_node*) malloc(sizeof(bh_oct_tree_node));
			if (new_inner_node == 0) {
				fprintf(stderr, "Error: Could not allocate inner node.\n");
				exit(1);
			}

			clear_children(new_inner_node);

			// Calculate the position for the new region based on both points
			// x_new = root_x - (0.5 * r) + offset_x
			// ... same for y and z
			new_inner_node->cofm.x = (root->cofm.x - half_r) + offset.x;
			new_inner_node->cofm.y = (root->cofm.y - half_r) + offset.y;
		  new_inner_node->cofm.z = (root->cofm.z - half_r) + offset.z;
			new_inner_node->mass = 0.0;
			new_inner_node->type = bhNonLeafNode;
			new_inner_node->id = -1;
			
			insert_point(new_inner_node, p, half_r);
			insert_point(new_inner_node, child, half_r);

			root->children[space] = new_inner_node;
		} else {
			insert_point(child, p, half_r);
		}
	}
}

void free_tree(bh_oct_tree_node *root) {
	int i;
	for (i = 0; i < 8; i++) {
		if (root->children[i] != 0 && root->children[i]->type == bhNonLeafNode) {
			free_tree(root->children[i]);
			root->children[i] = 0;
		}
	}

	free(root);
}

void compute_center_of_mass(bh_oct_tree_node *root) {
	int i = 0;
	int j = 0;
	float mass;
	vec3d cofm;
	vec3d cofm_child;
	bh_oct_tree_node * child;

	mass = 0.0;
	cofm.x = 0.0;
	cofm.y = 0.0; 
	cofm.z = 0.0;

	for (i = 0; i < 8; i++) {
		child = root->children[i];
		if (child != 0) {
			// compact child nodes for speed
			if (i != j) {
				root->children[j] = root->children[i];
				root->children[i] = 0;
			}

			j++;

			// If non leave node need to traverse children:
			if (child->type == bhNonLeafNode) {
				// summarize children
				compute_center_of_mass(child);
			} else {
				points_sorted[sortidx++] = child; // insert this point in sorted order
			}

			mass += child->mass;

			cofm_child.x = child->cofm.x * child->mass; // r*m
			cofm_child.y = child->cofm.y * child->mass; // r*m
			cofm_child.z = child->cofm.z * child->mass; // r*m

			cofm.x = cofm.x + cofm_child.x;
			cofm.y = cofm.y + cofm_child.y;
			cofm.z = cofm.z + cofm_child.z;

		}
	}

	cofm.x = cofm.x  * (1.0 / mass);
	cofm.y = cofm.y  * (1.0 / mass);
	cofm.z = cofm.z  * (1.0 / mass);

	root->cofm =cofm;
	root->mass = mass;
}

void advance_point(bh_oct_tree_node * p, float dthf, float dtime) {
	vec3d delta_v;
	vec3d velh;
	vec3d velh_by_dt;
	//vec3d delta_p;

	delta_v.x =  p->acc.x * dthf;
	delta_v.y =  p->acc.y * dthf;
	delta_v.z =  p->acc.z * dthf;

	velh.x = p->vel.x + delta_v.x;
	velh.y = p->vel.y + delta_v.y;
	velh.z = p->vel.z + delta_v.z;

	velh_by_dt.x = velh.x * dtime;
	velh_by_dt.y = velh.y * dtime;
	velh_by_dt.z = velh.z * dtime;

	p->cofm.x = p->cofm.x + velh_by_dt.x;
	p->cofm.y = p->cofm.y + velh_by_dt.y;
	p->cofm.z = p->cofm.z + velh_by_dt.z;

	p->vel.x = velh.x + delta_v.x;
	p->vel.y = velh.y + delta_v.y;
	p->vel.z = velh.z + delta_v.z;
}

int main(int argc, char **argv) {
	int t, i, j;
	cudaError_t e;
	
	TIME_START(overall);
	TIME_START(read_input);

	read_input(argc, argv);

	TIME_END(read_input);

	printf("Configuration: nbodies = %d, ntimesteps = %d, dtime = %lf eps = %lf, tol = %lf\n", nbodies, ntimesteps, dtime, eps, tol);

	for (t = 0; t < ntimesteps; t++) {
		printf("Time step %d:\n", t);
	 
		TIME_START(compute_bb);

		compute_bounding_box();

		TIME_END(compute_bb);
		TIME_START(build_tree);

		SAFE_MALLOC(root, sizeof(bh_oct_tree_node));
		
		root->id = -1;
		root->type = bhNonLeafNode;
		root->mass = 0.0;

		clear_children(root);
		
		// Root is never a leaf
		root->cofm = center;

		for (i = 0; i < nbodies; i++) {
			insert_point(root, &(points[i]), diameter * 0.5);
		}

		TIME_END(build_tree);
		TIME_START(compute_cofm);

		// Summarize nodes
		sortidx = 0;
		compute_center_of_mass(root);

		TIME_END(compute_cofm);
		TIME_START(force_calc);
		
		// Construct a blocked tree:
		TIME_START(build_tree_gpu);

		h_gpu_root = build_gpu_tree(root);
		if(h_gpu_root == 0) { 
			fprintf(stderr, "Error: Could not allocate GPU tree.");
			exit(1);
		}
		
		TIME_END(build_tree_gpu);

		//give enough space to theads
		CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitStackSize, h_gpu_root->depth*8*20*sizeof(int)));

		TIME_START(init_kernel);
		init_kernel<<<1,1>>>();
		cudaThreadSynchronize();		
		e = cudaGetLastError();
		if(e != cudaSuccess) {
			fprintf(stderr, "Error: init_kernel failed with error: %s\n", cudaGetErrorString(e));
			exit(1);
		}
		TIME_END(init_kernel);

		//printf("Tree depth = %d, nnodes = %d\n", h_gpu_root->depth, h_gpu_root->nnodes);
		TIME_START(gpu_copy_tree_to);
		
		allocate_gpu_tree_device(h_gpu_root, &d_gpu_root);		
	  copy_gpu_tree_to_device(h_gpu_root, d_gpu_root);

		// **** POINTER TABLE **** //
		
		dim3 blocks(NUM_THREAD_BLOCKS);
		dim3 tpb(THREADS_PER_BLOCK);
		
		TIME_START(pointer_table);
		
		hash_table_t ptable = allocate_ptrtab(d_gpu_root->nnodes * 8);				
		init_ptrtab<<<blocks,tpb>>>(ptable);
		cudaThreadSynchronize();		
		e = cudaGetLastError();
		if(e != cudaSuccess) {
			fprintf(stderr, "Error: init_ptrtab failed with error: %s\n", cudaGetErrorString(e));
			exit(1);
		}

		fill_ptrtab<<<blocks,tpb>>>(ptable, *d_gpu_root);
		cudaThreadSynchronize();				
		e = cudaGetLastError();
		if(e != cudaSuccess) {
			fprintf(stderr, "Error: fill_ptrtab failed with error: %s\n", cudaGetErrorString(e));
			exit(1);
		}

		TIME_END(pointer_table);
		
   	// **** END POINTER TABLE **** //

		
		bh_kernel_params params;
		params.root = *d_gpu_root; // tree root passed by value
		params.itolsq = inv_tol_squared;
		params.step = t;
		params.dthf = half_dtime;
		params.epssq = eps_squared;
		params.nbodies = nbodies;
		params.ptrtab_points_sorted = ptable;
		params.size = diameter;
		if(sort_flag)
			params.h_points_sorted = points_sorted;
		else {
			for(i = 0; i < nbodies; i++) {
				bh_oct_tree_node* tmp = points_unsorted[i];
				int r = rand() % nbodies;
				points_unsorted[i] = points_unsorted[r];
				points_unsorted[r] = tmp;
			}
			params.h_points_sorted = points_unsorted;						
		}

		allocate_kernel_params(&params);		
		copy_kernel_params_to_device(params);


		TIME_END(gpu_copy_tree_to);

		// **** KERNEL **** //

		TIME_START(kernel);
		compute_force_gpu<<<blocks, tpb>>>(params);
   	cudaThreadSynchronize();
		 
		e = cudaGetLastError();
		if(e != cudaSuccess) {
			fprintf(stderr, "Error: compute_force_gpu failed with error: %s\n", cudaGetErrorString(e));
			exit(1);
		}

		TIME_END(kernel);

		// **** END KERNEL **** //
		
		TIME_START(gpu_copy_tree_from);
		copy_gpu_tree_to_host(h_gpu_root, d_gpu_root);
    
		#ifdef TRACK_TRAVERSALS
		unsigned long long sum_nodes_accessed = 0;
		int *max_traversal_size_per_warp;
		SAFE_CALLOC(max_traversal_size_per_warp, nbodies,sizeof(int)); // waste memeory...
		#endif
    for(i = 0; i < h_gpu_root->nnodes; i++) {
      bh_oct_tree_node * n = h_gpu_root->nodes3[i].cpu_addr; // gets CPU node associated with GPU node i                                                                                                 
      // copy all values (Most values are not changed in recursive loop, only copy what we need.)                               

      n->vel = h_gpu_root->nodes3[i].vel;
      n->acc = h_gpu_root->nodes3[i].acc;
			#ifdef TRACK_TRAVERSALS
			if(n->type == bhLeafNode) {
				n->nodes_accessed = h_gpu_root->nodes3[i].nodes_accessed;
				sum_nodes_accessed += h_gpu_root->nodes3[i].nodes_accessed;
			}
			#endif
      // no child updates, assume tree structure is not changed                             
    }

		#ifdef TRACK_TRAVERSALS
		bh_oct_tree_node **pts = (sort_flag) ? points_sorted : points_unsorted;
		for(i = 0; i < nbodies + (nbodies % 32); i+=32) {
		  int na = pts[i]->nodes_accessed;
		  for(j = i + 1; j < i + 32 && j < nbodies; j++) {
		    if(pts[j]->nodes_accessed > na)
			   na = pts[j]->nodes_accessed;
	    }
			//printf("nodes warp %d: %d\n", i/32, na);
		}

        if (warp_flag) {
            int maximum = 0, all = 0, j = 0;
            unsigned long long maximum_sum = 0, all_sum = 0;
            bh_oct_tree_node **re_sorted = (sort_flag) ? points_sorted : points_unsorted;
            for(i = 0; i < nbodies + (nbodies % 32); i+=32) {
                maximum = re_sorted[i]->nodes_accessed;
                all = re_sorted[i]->nodes_accessed;
                for(j = i + 1; j < i + 32 && j < nbodies; j++) {
                    if(re_sorted[j]->nodes_accessed > maximum)
                        maximum = re_sorted[j]->nodes_accessed;
                    all += re_sorted[j]->nodes_accessed;                                                               }
                printf("%d\n", maximum);                                                                               maximum_sum += maximum;
                all_sum += all;
            }
//            delete re_sorted;
        }

		printf("total nodes: %llu\n", sum_nodes_accessed);
		printf("avg nodes: %f\n", (float)sum_nodes_accessed/nbodies);
		#endif

    free_gpu_tree_device(d_gpu_root);
    free_kernel_params_device(params);
    
		free_ptrtab(ptable);
		free_gpu_tree(h_gpu_root);
		TIME_END(gpu_copy_tree_from);

		TIME_END(force_calc);

		TIME_START(adv_point);

		for (i = 0; i < nbodies; i++) {
			advance_point(points_sorted[i], half_dtime, dtime);
		}
		
		TIME_END(adv_point);

		TIME_ELAPSED_PRINT(build_tree, stdout);
		TIME_ELAPSED_PRINT(compute_bb, stdout);
		TIME_ELAPSED_PRINT(compute_cofm, stdout);
		TIME_ELAPSED_PRINT(force_calc, stdout);
		TIME_ELAPSED_PRINT(init_kernel, stdout);
		TIME_ELAPSED_PRINT(gpu_copy_tree_to, stdout);
		TIME_ELAPSED_PRINT(kernel, stdout);
		TIME_ELAPSED_PRINT(gpu_copy_tree_from, stdout);
		TIME_ELAPSED_PRINT(build_tree_gpu, stdout);
		TIME_ELAPSED_PRINT(pointer_table, stdout);
		TIME_ELAPSED_PRINT(adv_point, stdout);
		
		// Cleanup before the next run
		free_tree(root);
		root = 0;
	}

	TIME_END(overall);

	// print the position of all points:
	if(verbose_flag) {
		for (i = 0; i < nbodies; i=i+10000) {
			bh_oct_tree_node * p = (points_sorted[i]);
			printf("%d %d: %1.4f %1.4f %1.4f\n", i, p->id, p->cofm.x, p->cofm.y, p->cofm.z);
		}
	}

	TIME_ELAPSED_PRINT(read_input, stdout);
	TIME_ELAPSED_PRINT(overall, stdout);

	// all other resources will be freed!

	return 0;
}

void print_oct_tree(bh_oct_tree_node * node, int child_pos, int level) {
	int i;
	for(i = 0; i < level; i++)
		printf("-");

  printf("Node %d - pos = %d\n", node->id, child_pos);
	for(i = 0; i < 8; i++)
		if(node->children[i] != NULL)
			print_oct_tree(node->children[i], i, level+1);

}

/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef BH_H_
#define BH_H_

#include <cuda.h>

typedef struct _vec3d {
	float x;
	float y;
	float z;
} vec3d;

typedef enum _bh_node_type {
	bhLeafNode,
	bhNonLeafNode
} bh_node_type;

typedef struct _bh_oct_tree_node {
	bh_node_type type;
	float mass;
	int id;
	int depth;
	int pre_id;
	vec3d cofm; /* center of mass */
	vec3d vel; /* current velocity */
	vec3d acc; /* current acceleration */
	struct _bh_oct_tree_node * children[8]; /* pointers to child nodes */
	#ifdef TRACK_TRAVERSALS
	int nodes_accessed;
	#endif
} bh_oct_tree_node;

#define clear_children(n) { (n)->children[0] = 0; (n)->children[1] = 0; (n)->children[2] = 0; (n)->children[3] = 0; \
		(n)->children[4] = 0; (n)->children[5] = 0; (n)->children[6] = 0; (n)->children[7] = 0; }

void print_oct_tree(bh_oct_tree_node * root, int n, int level);

#endif /* BH_H_ */

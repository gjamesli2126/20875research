/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "common.h"

extern unsigned int npoints;
extern unsigned int nthreads;

extern unsigned int K;
extern unsigned int max_depth;
extern unsigned int nnodes;
extern unsigned int avgnodes;

extern Node* tree;
extern DataPoint *points;
extern ClusterPoint* clusters;

static int sort_split;


void read_input(int argc, char **argv);
Node* construct_tree(ClusterPoint *clusters, int start_idx, int end_idx, int depth, Node* parent);
void deconstruct_tree(Node* root);
static int cmpfn_float(const void *a, const void *b);
void PrintClusters(FILE* fp);

#ifdef METRICS
extern int splice_depth;	
extern std::vector<Node*> subtrees;
class subtreeStats
{
public:
	long int footprint;
	int numnodes;
	subtreeStats(){footprint=0;numnodes=0;}
};
extern std::vector<subtreeStats> subtreeStatsList;

void printLoadDistribution(bool iterative);
void getSubtreeStats(Node* ver, subtreeStats* stats);
#endif

#endif
// end of __FUNCTIONS_H__

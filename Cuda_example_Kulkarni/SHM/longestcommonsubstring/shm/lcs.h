/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef LCS_H_
#define LCS_H_

#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>

#define MAX_CHAR 256
using namespace std;

struct SuffixTreeNode {
		struct SuffixTreeNode *children[MAX_CHAR];

		struct SuffixTreeNode *suffixLink;

		int start;
		int *end;

		int suffixIndex;

		int labelHeight;
};

typedef struct SuffixTreeNode Node;

//Globals
string text; //Input string
Node *root = NULL; //Pointer to root node

Node *lastNewNode = NULL;
Node *activeNode = NULL;

int activeEdge = -1;
int activeLength = 0;

int remainingSuffixCount = 0;
int leafEnd = -1;
int *rootEnd = NULL;
int *splitEnd = NULL;
int size = -1; //Length of input string
int size1 = 0; //Size of 1st string
int depth = 0; //depth of the suffix tree
int limit = 2; //recursion unroll limit
int nthread = 1; //number of threads
//Function declarations
Node *newNode(int start, int *end);
int edgeLength(Node *n);
int walkDown(Node *currNode);
void extendSuffixTree(int pos);
void setSuffixIndexByDFS(Node *n, int labelHeight, int d);
void freeSuffixTreeByPostOrder(Node *n);
void buildSuffixTree();
void dolevel(Node *n, int depth, int limit, vector<Node*>& list);
void doTraversal(Node *n, int* maxHeight, int* substringStartIndex);
string getLongestCommonSubstring();
double elapsedTime(struct timespec start, struct timespec finish);
#endif

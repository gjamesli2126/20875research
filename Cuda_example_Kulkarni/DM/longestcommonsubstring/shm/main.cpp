/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<boost/graph/parallel/algorithm.hpp>
#include "main.h"

using namespace boost;
using boost::graph::distributed::mpi_process_group;


Node *newNode(int start, int *end)
{
		Node *node =(Node*) malloc(sizeof(Node));
		for (int i = 0; i < MAX_CHAR; i++) node->children[i] = NULL;

		node->suffixLink = root;
		node->start = start;
		node->end = end;

		node->suffixIndex = -1;
		return node;
}

int edgeLength(Node *n) 
{
		if(n == root)
				return 0;
		return *(n->end) - (n->start) + 1;
}

int walkDown(Node *currNode)
{
		if (activeLength >= edgeLength(currNode))
		{
				activeEdge += edgeLength(currNode);
				activeLength -= edgeLength(currNode);
				activeNode = currNode;
				return 1;
		}
		return 0;
}

void extendSuffixTree(int pos)
{
		leafEnd = pos;

		remainingSuffixCount++;

		lastNewNode = NULL;

		while(remainingSuffixCount > 0) {

				if (activeLength == 0)
						activeEdge = pos; 

				if (activeNode->children[text.at(activeEdge)] == NULL)
				{
						activeNode->children[text.at(activeEdge)] = newNode(pos, &leafEnd);

						if (lastNewNode != NULL)
						{
								lastNewNode->suffixLink = activeNode;
								lastNewNode = NULL;
						}
				}
				else
				{
						Node *next = activeNode->children[text.at(activeEdge)];
						if (walkDown(next))
						{
								continue;
						}
						if (text.at(next->start + activeLength) == text.at(pos))
						{
								if(lastNewNode != NULL && activeNode != root)
								{
										lastNewNode->suffixLink = activeNode;
										lastNewNode = NULL;
								}

								activeLength++;
								break;
						}


						splitEnd = (int*) malloc(sizeof(int));
						*splitEnd = next->start + activeLength - 1;

						Node *split = newNode(next->start, splitEnd);
						activeNode->children[text.at(activeEdge)] = split;

						split->children[text.at(pos)] = newNode(pos, &leafEnd);
						next->start += activeLength;
						split->children[text.at(next->start)] = next;


						if (lastNewNode != NULL)
						{
								lastNewNode->suffixLink = split;
						}


						lastNewNode = split;
				}

				remainingSuffixCount--;
				if (activeNode == root && activeLength > 0) 
				{
						activeLength--;
						activeEdge = pos - remainingSuffixCount + 1;
				}
				else if (activeNode != root) 
				{
						activeNode = activeNode->suffixLink;
				}
		}
}

void setSuffixIndexByDFS(Node *n, int labelHeight, int d)
{
		if (n == NULL)  return;
		
		depth = (d > depth)? d : depth;
		n->labelHeight = labelHeight;
		
		int leaf = 1;
		for (int i = 0; i < MAX_CHAR; i++)
		{
				if (n->children[i] != NULL)
				{
						leaf = 0;
						setSuffixIndexByDFS(n->children[i], labelHeight + edgeLength(n->children[i]), d+1);
				}
		}
		if (leaf == 1)
		{
				for(int i= n->start; i<= *(n->end); i++)
				{
						if(text.at(i) == '#')
						{
								n->end = (int*) malloc(sizeof(int));
								*(n->end) = i;
						}
				}
				n->suffixIndex = size - labelHeight;
				if(n->suffixIndex > -1 && n->suffixIndex < size1) n->suffixIndex = -2;
				else if(n->suffixIndex >= size1) n->suffixIndex = -3;
		}
}

void freeSuffixTreeByPostOrder(Node *n)
{
		if (n == NULL) return;
		for (int i = 0; i < MAX_CHAR; i++)
		{
				if (n->children[i] != NULL)
				{
						freeSuffixTreeByPostOrder(n->children[i]);
				}
		}
		if (n->suffixIndex == -1)
				free(n->end);
		free(n);
}

void buildSuffixTree()
{
		size = text.length();
		rootEnd = (int*) malloc(sizeof(int));
		*rootEnd = - 1;

		root = newNode(-1, rootEnd);

		activeNode = root; 
		for (int i=0; i<size; i++) extendSuffixTree(i);
		int labelHeight = 0;
		setSuffixIndexByDFS(root, labelHeight, 0);
		cout << "Depth of the Suffix Tree : " << depth << endl;
}

void doTraversal(Node *n, int* maxHeight, int* substringStartIndex, int* ret)
{
		//-1 : Internal Node
		//-2 : Suffix of X
		//-3 : Suffix of Y
		//-4 : Suffix of X and Y
		if(n == NULL) return;

		for (int i = 0; i < MAX_CHAR; i++)
		{
				if(n->children[i] != NULL)
				{
						int r = -1;
						doTraversal(n->children[i], maxHeight, substringStartIndex, &r);

						if(n->suffixIndex == -1) n->suffixIndex = r;
						else if((n->suffixIndex == -2 && r == -3) || (n->suffixIndex == -3 && r == -2))
						{
								n->suffixIndex = -4;
								if(*maxHeight < n->labelHeight)
								{
										*maxHeight = n->labelHeight;
										*substringStartIndex = *(n->end) - n->labelHeight + 1;
								}
						}
				}
		}
		*ret = n->suffixIndex;
		return;
}

void dolevel(Node *n, int depth, int limit, vector<Node*>& list)
{
		if(depth > limit || n == NULL) return;
		
		list.push_back(n);

		for (int i = 0; i < MAX_CHAR; i++) dolevel(n->children[i], depth+1, limit, list);
		
}

string getLongestCommonSubstring(int procRank, int numProcs, mpi::communicator& world)
{
		vector<Node*> worklist;
		dolevel(root, 0, limit, worklist);
		
		int i = 0;
		int N = worklist.size();
		int numPoints = N;
		int startIndex = procRank * (numPoints/numProcs);
		int endIndex = (procRank == (numProcs - 1))?numPoints:((procRank+1) * (numPoints/numProcs));
		if(numPoints < numProcs)
		{
			if(procRank >= numPoints)
			{
				startIndex = 0;
				endIndex = 0;
			}
			else
			{
				startIndex = procRank;
				endIndex = procRank+1;
			}
		}

		int ret = -1;
		int* maxH = (int*)malloc(sizeof(int)*(endIndex-startIndex)); memset(maxH, 0, sizeof(int)*(endIndex-startIndex));
		int* substringStartIn = (int*)malloc(sizeof(int)*(endIndex-startIndex)); memset(substringStartIn, 0, sizeof(int)*(endIndex-startIndex));
		
		int maxHeight = 0, maxHeightGlobal;
		int substringStartIndex = 0, substringStartIndexGlobal=-1, tmpIndexGlobal;
		int resArray[2];
		
		for(i = 0; i < (endIndex-startIndex); i++)
		{
				ret = -1;
				doTraversal(worklist[i], maxH+i, substringStartIn+i, &ret);
				if (maxHeight < maxH[i])
				{
						maxHeight = maxH[i];
						substringStartIndex = substringStartIn[i];
				}
				
		}
		
		all_reduce(world,maxHeight, maxHeightGlobal, boost::parallel::maximum<int>());
		if(maxHeightGlobal == maxHeight)
			substringStartIndexGlobal = substringStartIndex;
		all_reduce(world,substringStartIndexGlobal, tmpIndexGlobal, boost::parallel::maximum<int>());
			
		
		int k; ostringstream oss;
		for (k=0; k<maxHeight; k++) oss << text.at(k + tmpIndexGlobal);
		if(k == 0) oss << "";

		return oss.str();
}

double elapsedTime(struct timespec start, struct timespec finish)
{
		double elapsed;
		if(start.tv_nsec > finish.tv_nsec){
				elapsed = (finish.tv_sec - start.tv_sec - 1);
				elapsed += (1000000000 + finish.tv_nsec - start.tv_nsec) / 1000000000.0;
		}
		else{
				elapsed = (finish.tv_sec - start.tv_sec);
				elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
		}

		return elapsed;

}

int main(int argc, char *argv[])
{
	boost::mpi::environment env(argc, argv);
	mpi::communicator world;
	mpi_process_group pg;

	int procRank = process_id(pg);
	int numProcs = num_processes(pg);

	if (argc < 4) { cout << "Usage: ./run <LIMIT>  <INPUTX-PATH> <INPUTY-PATH>" << endl; return -1; }
		
		limit = atoi(argv[1]);

		struct timespec start, finish;
		double elapsed;

		string filenameX(argv[2]); filenameX = "../input/" + filenameX;
		string filenameY(argv[3]); filenameY = "../input/" + filenameY;

		ifstream fileX(filenameX);	
		ifstream fileY(filenameY);	
		string X((istreambuf_iterator<char>(fileX)), istreambuf_iterator<char>());
		string Y((istreambuf_iterator<char>(fileY)), istreambuf_iterator<char>()); 
		
		text = X + "#" + Y + "$";
		size1 = X.length()+1;

		buildSuffixTree();

		clock_gettime(CLOCK_MONOTONIC, &start);
		string lcp = getLongestCommonSubstring(procRank, numProcs, world);
		clock_gettime(CLOCK_MONOTONIC, &finish);
		elapsed = elapsedTime(start, finish);
		double maxElapsedTime;
		reduce(world,elapsed, maxElapsedTime, boost::parallel::maximum<double>(),0);
#ifdef DEBUG
		if(procRank == 0)
		{
			if(lcp.length() > 0) cout << "Longest Common Substring in " << X << "and" << Y << " is: " << lcp << " , length of : " << lcp.length() << endl;
			else cout << "Longest Common Substring in " << X << " and " << Y << " is: No common substring" << endl; 
		}
#endif

		if(procRank == 0)
		{
			cout << "X length : " << X.length() << endl;
			cout << "Y length : " << Y.length() << endl;
			cout << "LCP length : " << lcp.length() << endl;
			cout << "Elapsed Time : " << maxElapsedTime << " sec" << endl;
		}


		freeSuffixTreeByPostOrder(root);

		return 0;
}

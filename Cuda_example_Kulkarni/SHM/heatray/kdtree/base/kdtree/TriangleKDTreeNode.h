/*

   Filename : TriangleKDtreeNode.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Triangle-based kd-tree node. 

   Change List:

      - 01/26/2011  - Created (Cody White)

*/

#pragma once

#include <shared/Ray.h>
#include <shared/Box.h>
#include <math/Vector.h>
#include <gfx/Triangle.h>

/**
  * KDTree node.
  */
class TriangleKDTreeNode {
public:
	TriangleKDTreeNode();
	~TriangleKDTreeNode();

	float getArea (void) const;
	int intersectCount (const std::vector <gfx::Triangle *> &list) const;
	bool checkRay (const Ray &ray, float &time) const;
	bool isLeaf (void) const;
	bool insertTriangle (gfx::Triangle *triangle);

	std::vector <gfx::Triangle *> triangles;		// List of triangles contained within this node.
	TriangleKDTreeNode *left;							// Left child of this node.
	TriangleKDTreeNode *right;							// Right child of this node.
	Box box;										// Bounding box for this node.

#ifdef METRICS
	int numpointsvisited;
#endif
#ifdef TRACK_TRAVERSALS
	int id;
private:
	static int next_id;
#endif
};

#ifdef METRICS
class subtreeStats
{
public:
	long int footprint;
	int numnodes;
	subtreeStats(){footprint=0;numnodes=0;}
};

extern int numberOfTraversals;
extern int splice_depth;
extern int max_depth;
extern std::vector<TriangleKDTreeNode*> subtrees;
extern std::vector<subtreeStats> subtreeStatsList;

void printLoadDistribution(bool iterative);
void getSubtreeStats(TriangleKDTreeNode* ver, subtreeStats* stats, int DOR);
#endif

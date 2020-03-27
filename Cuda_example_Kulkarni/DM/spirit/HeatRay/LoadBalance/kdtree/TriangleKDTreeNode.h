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
using namespace gfx;

class TriangleKDTreeNode {
public:
	TriangleKDTreeNode();
	~TriangleKDTreeNode();

	float getArea (void) const;
	int intersectCount (const std::vector <gfx::Triangle *> &list) const;
	bool checkRay (const Ray &ray, float &time) const;
	bool isLeaf (void) const;
	bool insertTriangle (gfx::Triangle *triangle);

	std::vector<gfx::Triangle> triangles;
	Box box;						// Bounding box for this node.
};



/*
 * TriangleKDTreeNode.cpp
 *
 *  Created on: Nov 5, 2012
 *      Author: yjo
 */

#include "kdtree/TriangleKDTreeNode.h"

/**
 * Default constructor.
 */
TriangleKDTreeNode::TriangleKDTreeNode (void)
{
}

/**
 * Destructor.
 */
TriangleKDTreeNode::~TriangleKDTreeNode (void)
{
	triangles.clear ();
}

/**
 * Return the area of the node.
 */
float TriangleKDTreeNode::getArea (void) const
{
	return box.getArea ();
}

/**
 * Get the amount of triangles from a passed in list
 * that would be contained within this node.
 * @param list List of triangles to test with.
 */
int TriangleKDTreeNode::intersectCount (const std::vector <gfx::Triangle *> &list) const
{
	int count = 0;
	for (size_t i = 0; i < list.size (); ++i)
	{
		count += (box.testTriangle (list[i]) ? 1 : 0);
	}

	return count;
}

/**
 * Check a ray for collision with this node.
 * @param ray Ray to check with.
 * @param time Time value returned if the ray hit.
 */
bool TriangleKDTreeNode::checkRay (const Ray &ray, float &time) const
{
	return box.testRay (ray, time);
}



/**
 * Insert a triangle into this node.
 * @param triangle Triangle to insert.
 */
bool TriangleKDTreeNode::insertTriangle (gfx::Triangle *triangle)
{
	if (box.testTriangle (triangle))
	{
		triangles.push_back (*triangle);
		return true;
	}

	return false;
}

/*
 * TriangleKDTreeNode.cpp
 *
 *  Created on: Nov 5, 2012
 *      Author: yjo
 */

#include "kdtree/TriangleKDTreeNode.h"

#ifdef TRACK_TRAVERSALS
int TriangleKDTreeNode::next_id = 0;
#endif

/**
 * Default constructor.
 */
TriangleKDTreeNode::TriangleKDTreeNode (void)
{
	left = NULL;
	right = NULL;
#ifdef TRACK_TRAVERSALS
	id = next_id++;
#endif

#ifdef METRICS
	numpointsvisited=0;
#endif
}

/**
 * Destructor.
 */
TriangleKDTreeNode::~TriangleKDTreeNode (void)
{
	triangles.clear ();
	left = NULL;
	right = NULL;
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
	//printf("testing ray:origin=(%f %f %f) direction=(%f %f %f) with box. min=(%f %f %f) max=(%f %f %f)\n",ray.origin[0],ray.origin[1],ray.origin[2],ray.direction[0],ray.direction[1],ray.direction[2], box.min[0],box.min[1],box.min[2],box.max[0], box.max[1],box.max[2]);
	return box.testRay (ray, time);

}

/**
 * Return leaf or not.
 */
bool TriangleKDTreeNode::isLeaf (void) const
{
	return (left == NULL && right == NULL);
}

/**
 * Insert a triangle into this node.
 * @param triangle Triangle to insert.
 */
bool TriangleKDTreeNode::insertTriangle (gfx::Triangle *triangle)
{
	if (box.testTriangle (triangle))
	{
		triangles.push_back (triangle);
		return true;
	}

	return false;
}

#ifdef METRICS
void printLoadDistribution(bool iterative)
{
	if(!iterative)
		printf("num bottom subtrees %d\n",subtrees.size());
	std::vector<TriangleKDTreeNode*>::iterator iter = subtrees.begin();
	for(;iter != subtrees.end();iter++)
	{
		long int num_vertices=0, footprint=0;
		subtreeStats stats;
		int DOR=0;
		if((*iter)->id != 0)
			DOR=-1;
		getSubtreeStats(*iter, &stats, DOR);
		if(!iterative)
			printf("id %p num_vertices %d footprint %ld\n", *iter, stats.numnodes, stats.footprint);
		else
			subtreeStatsList.push_back(stats);
	}
}

void getSubtreeStats(TriangleKDTreeNode* ver, subtreeStats* stats, int DOR)
{
		
		if(DOR == (splice_depth))
		{
			return;
		}
		if(DOR >= 0)
		{
			DOR +=1;
		}	
		stats->numnodes += 1;
		stats->footprint += ver->numpointsvisited;

		if((ver->left == NULL) && (ver->right==NULL))
		{
			return;
		}

		if(ver->left)
		{
			getSubtreeStats(ver->left,stats, DOR);
		}
		if(ver->right)
		{
			getSubtreeStats(ver->right,stats, DOR);
		}
}
#endif

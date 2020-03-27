/*

   Filename : TriangleKDtree.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Triangle-based kd-tree for ray tracing. 

   Change List:

      - 12/22/2009  - Created (Cody White)

*/

#pragma once

#include "gfx/gfx.h"

#include <kdtree/TriangleKDTreeNode.h>
#include <shared/IntersectInfo.h>
#include <shared/Ray.h>
#include <gfx/Triangle.h>
#include <gfx/Mesh.h>
#include <list>
#include <vector>
#include <iostream>
#include <set>
#include<map>
#if defined(PROFILE) || defined(TREE_PROFILE)
#include "profilers.h"
#endif
/**
  * KDTree class.
  */
class TriangleKDTree 
{
	public:
		/**
		  * Default constructor.
		  */
		TriangleKDTree (void)
		{
			m_root = NULL;
#ifdef PROFILE
			m_profiler = new CountProfiler(8);
#endif
		}

		/**
		  * Destructor.
		  */
		~TriangleKDTree (void)
		{
			clear ();
#ifdef PROFILE
			m_profiler->output();
			delete m_profiler;
#endif
		}


		void build (gfx::Mesh &mesh);


		void clear (void);
		void intersect (Ray &ray, IntersectInfo &info);
		void render (void) const;
		void renderBorders (void) const;

	private:
#if defined(PROFILE)
		CountProfiler *m_profiler;
#endif
#ifdef TREE_PROFILE
		void shapeR(const TriangleKDTreeNode *node, int depth, CountProfiler &profiler);
#endif

		/**
		  * Recursively build the tree.
		  * @param node TriangleKDTreeNode to split.
		  * @param current_depth Current depth in the tree.
		  */
		void buildR (TriangleKDTreeNode *&node, int current_depth);
		bool sahSplit (TriangleKDTreeNode *&node, int current_depth);
		bool medianSplit (TriangleKDTreeNode *&node, int depth);
		void renderBordersR (const TriangleKDTreeNode *node) const;
		void renderR (const TriangleKDTreeNode *node) const;
		void clearR (TriangleKDTreeNode *&node);
		bool split (TriangleKDTreeNode *&node, int depth);
		void intersectR (const TriangleKDTreeNode *node, Ray &ray, IntersectInfo &info);

		TriangleKDTreeNode *m_root;
		std::map<int,long int> numLeavesAtHeight;
};


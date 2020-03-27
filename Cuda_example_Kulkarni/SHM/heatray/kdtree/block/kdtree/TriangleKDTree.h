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
#ifdef TREE_PROFILE
#include "treeprofiler.h"
#endif
#include<fstream>

class Block;
class BlockStack;
class _Autotuner;

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
		}

		/**
		  * Destructor.
		  */
		~TriangleKDTree (void)
		{
			clear ();
		}


		void build (gfx::Mesh &mesh);
#ifdef TREE_PROFILE
		TreeProfiler m_profiler;
#endif


		void clear (void);
		void intersect (Ray &ray, IntersectInfo &info);
		void intersect_prologue (Ray &ray, IntersectInfo &info, Block *_block);
		void intersect_main (BlockStack *_stack);
		void intersect_mainAutotune (BlockStack *_stack, _Autotuner *_autotuner);

		void render (void) const;
		void renderBorders (void) const;
		void print_treetofile(std::ofstream& fp);
		void print_preorder(TriangleKDTreeNode* node, std::ofstream& fp);

	private:

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
		void intersectR_block(const TriangleKDTreeNode *node, BlockStack *_stack, int _depth);
		void intersectR_blockAutotune(const TriangleKDTreeNode *node, BlockStack *_stack, int _depth, _Autotuner *_autotuner);

		TriangleKDTreeNode *m_root;
};


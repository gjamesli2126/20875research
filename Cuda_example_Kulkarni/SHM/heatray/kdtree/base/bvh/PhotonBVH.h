/*

   Filename : PhotonBVH.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : BVH for photon mapping. 

   Change List:

      - 03/22/2011  - Created (Cody White)

*/

#pragma once

#include "gfx/gfx.h"

#include <bvh/PhotonBVHNode.h>
#include <shared/Photon.h>
#include <shared/SearchResult.h>
#include <map>
#include <queue>
#include <iostream>

class PhotonBVH
{
	public:

		/**
		  * Default constructor.
		  */
		PhotonBVH (void)
		{
			m_root = NULL;
		}

		/**
		  * Destructor.
		  */
		~PhotonBVH (void)
		{
			clear ();
		}


		void clear (void);

		bool build (std::vector <Photon > &photons);
		bool exists (void);
		void render (void) const;
		void renderTree (void) const;
#ifdef TRACK_TRAVERSALS
		int
#else
		void
#endif
		knn (math::Vector <float, 3> search_point, size_t max_count, size_t max_distance, std::vector <SearchResult > &results) const;

	private:

		/**
		  * Structure for KNN.
		  */
		struct KNNSearch
		{
			/**
			  * Default constructor.
			  */
			KNNSearch (void)
			{
				photon = NULL;
				node = NULL;
				distance = (float)0.0;
				is_node = false;
			}

			/**
			  * Copy constructor.
			  */
			KNNSearch (const KNNSearch &other)
			{
				*this = other;
			}

			/**
			  * Operator=.
			  */
			KNNSearch & operator= (const KNNSearch &other)
			{
				if (this != &other)
				{
					photon = other.photon;
					node = other.node;
					distance = other.distance;
					is_node = other.is_node;
				}

				return *this;
			}

			/**
			  * Operator<
			  */
			bool operator< (const KNNSearch &other) const
			{
				return (distance > other.distance);
			}

			// Member variables.
			Photon  *photon;
			PhotonBVHNode *node;
			bool is_node;
			float distance;
		};


		void clearR (PhotonBVHNode *&node);
		void buildR (PhotonBVHNode *&node);
		void renderR (const PhotonBVHNode *node) const;
		void renderTreeR (const PhotonBVHNode *node) const;

		// Member variables.
		PhotonBVHNode *m_root;	// Root of the tree.
		size_t m_data_size;			// Size of the data in the tree.
};


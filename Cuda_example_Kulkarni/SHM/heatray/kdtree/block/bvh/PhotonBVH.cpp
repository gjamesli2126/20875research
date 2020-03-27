/*

   Filename : PhotonBVH.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : BVH for photon mapping.

   Change List:

      - 03/22/2011  - Created (Cody White)

 */

#include <bvh/PhotonBVH.h>

#define MAX_PHOTONS 100



/**
 * Clear the tree.
 */
void PhotonBVH::clear (void)
{
	clearR (m_root);
}

/**
 * Build the tree.
 * @param photons List of photons to partition into the tree.
 */
bool PhotonBVH::build (std::vector <Photon > &photons)
{
	// Make sure the tree is empty and ready to go.
	// If not, clear it.
	if (m_root != NULL)
	{
		clear ();
	}

	m_root = new PhotonBVHNode;
	m_data_size = photons.size ();

	// Add the photons to the root node.
	for (size_t i = 0; i < photons.size (); ++i)
	{
		m_root->photons.push_back (photons[i]);
	}

	buildR (m_root);
	return true;
}

/**
 * Deterine if the BVH exists or not.
 */
bool PhotonBVH::exists (void)
{
	return (m_root != NULL);
}

/**
 * Render the photon map.
 */
void PhotonBVH::render (void) const
{
	renderR (m_root);
}

/**
 * Render the tree.
 */
void PhotonBVH::renderTree (void) const
{
	renderTreeR (m_root);
}

/**
 * Find the k-Nearest neighbors to a point.
 * @param search_point Point to search around.
 * @param max_count Maximum number of neighbors to return.
 * @param max_distance Maximum distance for a point to be from search_point.
 * @param results Results vector full of points.
 */
#ifdef TRACK_TRAVERSALS
int
#else
void
#endif
PhotonBVH::knn (math::Vector <float, 3> search_point, size_t max_count, size_t max_distance, std::vector <SearchResult > &results) const
{
	int cnt = 0;
	size_t max_dist = max_distance * max_distance;
	std::priority_queue <KNNSearch> queue;

	if (m_root == NULL)
	{
#ifdef TRACK_TRAVERSALS
		return cnt;
#else
		return;
#endif
	}

	KNNSearch top;
	size_t found = 0;

	// Push the root onto the queue.
	top.node = m_root;
	top.is_node = true;
	top.distance = m_root->box.calcDistanceSquared (search_point);
	queue.push (top);

	while (found < max_count && found < m_data_size)
	{
#ifdef TRACK_TRAVERSALS
		cnt++;
#endif
		// Get the top element of the queue.
		top = queue.top ();
		// Remove the element from the queue.
		queue.pop ();

		// If the top is a point, compare it to the max_distance and possibly add it to the results.
		if (top.is_node == false)
		{
			if (top.distance > max_dist)
			{
#ifdef TRACK_TRAVERSALS
				return cnt;
#else
				return;
#endif
			}

			SearchResult r;
			r.photon = top.photon;
			r.distance = sqrtf (top.distance);
			results.push_back (r);
			found++;
		}

		else
		{
			if (top.node != NULL)
			{
				if (top.node->left == NULL && top.node->right == NULL)
				{
					// This is a leaf node, add all of its points to the queue.
					KNNSearch p;
					for (size_t i = 0; i < top.node->photons.size (); ++i)
					{
						p.photon = &(top.node->photons[i]);
						p.distance = math::length2 (search_point - top.node->photons[i].position);
						queue.push (p);
					}
				}

				else
				{
					// Push the nodes onto the queue.
					KNNSearch n;
					n.is_node = true;
					if (top.node->left != NULL)
					{
						n.node = top.node->left;
						n.distance = top.node->left->box.calcDistanceSquared (search_point);
						queue.push (n);
					}

					if (top.node->right != NULL)
					{
						n.node = top.node->right;
						n.distance = top.node->right->box.calcDistanceSquared (search_point);
						queue.push (n);
					}
				}
			}
		}
	}
#ifdef TRACK_TRAVERSALS
	return cnt;
#endif
}




/**
 * Recursively clear the tree.
 * @param node Node to clear.
 */
void PhotonBVH::clearR (PhotonBVHNode *&node)
{
	if (node != NULL)
	{
		clearR (node->left);
		clearR (node->right);
	}

	delete node;
	node = NULL;
}

/**
 * Recursively build the tree using median split.
 * @param node Node to shrink and subdivide.
 */
void PhotonBVH::buildR (PhotonBVHNode *&node)
{
	if (node == NULL)
	{
		return;
	}

	// Determine the maximum and minimum point of the bounding box
	// to contain all of the photons.  This must be performed in
	// order to create a tightly fitting bounding box around the
	// photons.
	math::Vector <float, 3> max (-HUGE_VAL, -HUGE_VAL, -HUGE_VAL);
	math::Vector <float, 3> min (HUGE_VAL, HUGE_VAL, HUGE_VAL);
	for (size_t i = 0; i < node->photons.size (); ++i)
	{
		max = math::vectorMax (max, node->photons[i].position);
		min = math::vectorMin (min, node->photons[i].position);
	}

	// Create the tight-fitting box for these photons.
	node->box = Box (min, max);

	// Only split if the number of photons exceeds the maximum for a node.
	if (node->photons.size () > MAX_PHOTONS)
	{
		// Determine the longest axis to split along.
		float length = (float)0.0;
		int axis = 0;
		for (int i = 0; i < 3; ++i)
		{
			float tmp = node->box.max[i] - node->box.min[i];
			if (tmp > length)
			{
				length = tmp;
				axis = i;
			}
		}

		// Determine the middle point along this axis.
		std::multimap <float, float> map;
		for (size_t i = 0; i < node->photons.size (); ++i)
		{
			map.insert (std::make_pair (node->photons[i].position[axis], node->photons[i].position[axis]));
		}

		std::multimap <float, float>::iterator iter = map.begin ();
		for (size_t i = 0; i < map.size () * 0.5f; ++i)
		{
			iter++;
		}

		float split_plane = iter->first;
		map.clear ();

		// Sort the photons into the left and right subchild.
		node->left = new PhotonBVHNode;
		node->right = new PhotonBVHNode;

		for (size_t i = 0; i < node->photons.size (); ++i)
		{
			if (node->photons[i].position[axis] < split_plane)
			{
				node->left->photons.push_back (node->photons[i]);
			}
			else
			{
				node->right->photons.push_back (node->photons[i]);
			}
		}

		if (node->left->photons.size () == 0)
		{
			delete node->left;
			node->left = NULL;
			return;
		}

		if (node->right->photons.size () == 0)
		{
			delete node->right;
			node->right = NULL;
			return;
		}

		node->photons.clear ();

		// Recurse into the child nodes.
		buildR (node->left);
		buildR (node->right);
	}
}

/**
 * Recursively render the photon map.
 */
void PhotonBVH::renderR (const PhotonBVHNode *node) const
{
#ifdef USE_GL
	if (node != NULL)
	{
		glBegin (GL_POINTS);
		for (size_t i = 0; i < node->photons.size (); ++i)
		{
			glColor3f (node->photons[i].power[0], node->photons[i].power[1], node->photons[i].power[2]);
			glVertex3f (node->photons[i].position[0], node->photons[i].position[1], node->photons[i].position[2]);
		}
		glEnd ();

		/*glColor3f (1.0f, 1.0f, 1.0f);
				glBegin (GL_LINES);
					for (size_t i = 0; i < node->photons.size (); ++i)
					{
						glVertex3fv (node->photons[i].position.v);
						glVertex3fv ((node->photons[i].position + node->photons[i].normal * 0.1f).v);
					}
				glEnd ();*/

		renderR (node->left);
		renderR (node->right);
	}
#endif
}

/**
 * Recursively render the tree.
 */
void PhotonBVH::renderTreeR (const PhotonBVHNode *node) const
{
#ifdef USE_GL
	if (node != NULL)
	{
		math::vec3f max = node->box.max;
		math::vec3f min = node->box.min;

		glBegin (GL_LINES);
		glVertex3f (max.x (), max.y (), max.z ()); glVertex3f (min.x (), max.y (), max.z ());
		glVertex3f (min.x (), max.y (), max.z ()); glVertex3f (min.x (), min.y (), max.z ());
		glVertex3f (min.x (), min.y (), max.z ()); glVertex3f (max.x (), min.y (), max.z ());
		glVertex3f (max.x (), min.y (), max.z ()); glVertex3f (max.x (), max.y (), max.z ());

		glVertex3f (min.x (), min.y (), min.z ()); glVertex3f (max.x (), min.y (), min.z ());
		glVertex3f (max.x (), min.y (), min.z ()); glVertex3f (max.x (), max.y (), min.z ());
		glVertex3f (max.x (), max.y (), min.z ()); glVertex3f (min.x (), max.y (), min.z ());
		glVertex3f (min.x (), max.y (), min.z ()); glVertex3f (min.x (), min.y (), min.z ());

		glVertex3f (max.x (), max.y (), max.z ()); glVertex3f (max.x (), max.y (), min.z ());
		glVertex3f (min.x (), max.y (), max.z ()); glVertex3f (min.x (), max.y (), min.z ());
		glVertex3f (min.x (), min.y (), max.z ()); glVertex3f (min.x (), min.y (), min.z ());
		glVertex3f (max.x (), min.y (), max.z ()); glVertex3f (max.x (), min.y (), min.z ());
		glEnd ();

		renderTreeR (node->left);
		renderTreeR (node->right);
	}
#endif
}



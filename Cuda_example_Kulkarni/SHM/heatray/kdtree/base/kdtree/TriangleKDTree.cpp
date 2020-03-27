/*

   Filename : TriangleKDtree.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Triangle-based kd-tree for ray tracing.

   Change List:

      - 12/22/2009  - Created (Cody White)

 */


#include <kdtree/TriangleKDTree.h>
#include <assert.h>

typedef math::Vector <float, 3> kvec;

/**
 * Build the KDtree.
 * @param mesh Triangle mesh to use for building the tree.
 */
void TriangleKDTree::build (gfx::Mesh &mesh)
{
	kvec min (HUGE_VAL, HUGE_VAL, HUGE_VAL);
	kvec max (-HUGE_VAL, -HUGE_VAL, -HUGE_VAL);
	m_root = new TriangleKDTreeNode;

	// Find the maximum and minimum point;
	for (size_t t = 0; t < mesh.getNumMeshes (); ++t)
	{
		gfx::Mesh::MeshPiece *piece = mesh.getMesh (t);
		for (size_t i = 0; i < piece->triangles.size (); ++i)
		{
			for (int j = 0; j < 3; ++j)// Loop over triangle vertices
			{
				for (int k = 0; k < 3; ++k)// Loop over vertex indices.
				{
					max[k] = std::max (max[k], piece->triangles[i].m_vertices[j][k]);
					min[k] = std::min (min[k], piece->triangles[i].m_vertices[j][k]);
				}
			}

			piece->triangles[i].initializeIntersection ();
			m_root->triangles.push_back (&(piece->triangles[i]));
		}
	}

	m_root->box = Box (min, max);
	m_root->box.calcDimensions ();
	buildR (m_root, 0);
	printf("max depth of tree %d\n",max_depth);
	/*std::map<int,long int>::iterator it;
	for (it=numLeavesAtHeight.begin(); it!=numLeavesAtHeight.end(); ++it)
	{
		printf("Total Leaves At Height: %d: %ld\n",it->first,it->second);
	}*/

#ifdef TREE_PROFILE
	CountProfiler profiler(60);
	shapeR(m_root, 0, profiler);
	profiler.output();
#endif
}
#ifdef TREE_PROFILE
void TriangleKDTree::shapeR(const TriangleKDTreeNode *node, int depth, CountProfiler &profiler) {
	if (node == NULL) return;
	profiler.record(depth, 1);
	shapeR (node->left, depth + 1, profiler);
	shapeR (node->right, depth + 1, profiler);
}
#endif

/**
 * Clear the kdtree.
 */
void TriangleKDTree::clear (void)
{
	clearR (m_root);
	m_root = NULL;
}

/**
 * Intersect with the kdtree.
 * @param ray Ray to use for intersection tests.
 * @param info Intersection info struct to populate if a hit occurs.
 * @param shadow If this is a shadow ray, we can avoid some of the accuracy of an actual intersection to speed up this test.
 * @param tmin Minimum time to an intersection.
 */
void TriangleKDTree::intersect (Ray &ray, IntersectInfo &info)
{
	ray.inverse_direction[0] = (float)1.0 / ray.direction[0];
	ray.inverse_direction[1] = (float)1.0 / ray.direction[1];
	ray.inverse_direction[2] = (float)1.0 / ray.direction[2];
	ray.intersects = false;
	intersectR (m_root, ray, info);
}

/**
 * Render the data inside of the tree.
 */
void TriangleKDTree::render (void) const
{
	renderR (m_root);
}

/**
 * Render the borders of the tree.
 */
void TriangleKDTree::renderBorders (void) const
{
#ifdef USE_GL
	glPushAttrib (GL_LIGHTING_BIT | GL_CURRENT_BIT);
	glDisable (GL_LIGHTING);
	glColor3f (0.2f, 1.0f, 0.2f);
	renderBordersR (m_root);
	glPopAttrib ();
#endif
}


/**
 * Recursively build the tree.
 * @param node TriangleKDTreeNode to split.
 * @param current_depth Current depth in the tree.
 */
void TriangleKDTree::buildR (TriangleKDTreeNode *&node, int current_depth)
{
	if (node == NULL)
	{
		return;
	}

	if (split (node, current_depth))
	{
		/*if(node->left && node->right)
		{
			assert((node->left->triangles.size() > 0) && (node->right->triangles.size() > 0));
			printf("current_depth:%d leftPoints:%d rightPoints:%d selfpoints:%d leftBox min=(%f %f %f) leftBox max=(%f %f %f) rightBox min=(%f %f %f) rightBox max=(%f %f %f)\n",current_depth,((node->left)->triangles).size(),((node->right)->triangles).size(), node->triangles.size(),node->left->box.min[0],node->left->box.min[1],node->left->box.min[2],node->left->box.max[0],node->left->box.max[1],node->left->box.max[2],node->right->box.min[0],node->right->box.min[1],node->right->box.min[2],node->right->box.max[0],node->right->box.max[1],node->right->box.max[2]);
		}
		else if(node->left)
		{
			assert(node->left->triangles.size() > 0);
			printf("current_depth:%d leftPoints:%d rightPoints:0 selfpoints:%d leftBox min=(%f %f %f) leftBox max=(%f %f %f) rightBox min=(0.0 0.0 0.0) rightBox max=(0.0 0.0 0.0)\n",current_depth,((node->left)->triangles).size(),node->triangles.size(),node->left->box.min[0],node->left->box.min[1],node->left->box.min[2],node->left->box.max[0],node->left->box.max[1],node->left->box.max[2]);
		}
		else if(node->right)
		{
			assert(node->right->triangles.size() > 0);
			printf("current_depth:%d leftPoints:0 rightPoints:%d selfpoints:%d leftBox min=(0.0 0.0 0.0) leftBox max=(0.0 0.0 0.0) rightBox min=(%f %f %f) rightBox max=(%f %f %f)\n",current_depth,((node->right)->triangles).size(),node->triangles.size(),node->right->box.min[0],node->right->box.min[1],node->right->box.min[2],node->right->box.max[0],node->right->box.max[1],node->right->box.max[2]);
		}
		else if(!(node->left) && !(node->right))
			printf("ERROR\n");*/
			
		buildR (node->left, current_depth + 1);
		buildR (node->right, current_depth + 1);
	}
	else
	{
		assert((node->left == NULL) && (node->right == NULL));
		std::pair<std::map<int,long int>::iterator,bool> ret;
		ret = numLeavesAtHeight.insert(std::pair<int,long int>(current_depth,1) );
		if(ret.second==false)
			numLeavesAtHeight[current_depth] += 1;
	}
#ifdef METRICS
    	if (current_depth > max_depth)
        	max_depth = current_depth;
	if(current_depth == splice_depth)
		subtrees.push_back(node);
#endif
}

/**
 * Perform a SAH split.
 * @param node TriangleKDTreeNode to perform the splitting of.
 * @param current_depth Current depth in the tree.
 */
bool TriangleKDTree::sahSplit (TriangleKDTreeNode *&node, int current_depth)
{
	int split_axis = current_depth % 3;
	float best_split_plane = (float)0.0;
	float traversal_cost = (float)0.3;
	float intersect_cost = (float)1.0;
	float best_cost = (float)HUGE_VAL;
	TriangleKDTreeNode *left = new TriangleKDTreeNode;
	TriangleKDTreeNode *right = new TriangleKDTreeNode;
	std::set<float>  split_planes; // Set of split planes already tried.
	std::pair <std::set<float> ::iterator, bool> return_value;
	float parent_area = (float)1.0 / node->getArea ();

	// Determine the best split plane to use.
	// Loop over the triangles in this node, the split planes are selected as the
	// current split axis of each primitive.
	for (size_t i = 0; i < node->triangles.size (); ++i)
	{
		float split_plane = (float)HUGE_VAL;
		// Find the smallest of the vertices of this triangle.
		for (int j = 0; j < 3; ++j)
		{
			if (node->triangles[i]->m_vertices[j][split_axis] < split_plane)
			{
				// Make sure that this split plane lies inside of the node.  Because triangles can
				// span multiple nodes, this check is necessary.
				if (node->triangles[i]->m_vertices[j][split_axis] < node->box.max[split_axis] &&
						node->triangles[i]->m_vertices[j][split_axis] > node->box.min[split_axis])
				{
					split_plane = node->triangles[i]->m_vertices[j][split_axis];
				}
			}
		}

		if (split_plane != HUGE_VAL)
		{
			return_value = split_planes.insert (split_plane);
			if (return_value.second)
			{
				// We have not yet evaluated this split plane.
				// Make two new nodes that would be children of this node using the current split plane.
				left->box.min = node->box.min;
				left->box.max = node->box.max;
				left->box.max[split_axis] = split_plane;

				right->box.min = node->box.min;
				right->box.max = node->box.max;
				right->box.min[split_axis] = split_plane;

				left->box.calcDimensions ();
				right->box.calcDimensions ();

				int left_count  = left->intersectCount (node->triangles);
				int right_count = right->intersectCount (node->triangles);
				float left_prob = left->getArea () * parent_area;
				float right_prob = right->getArea () * parent_area;

				float cost = traversal_cost + intersect_cost * (left_prob * left_count + right_prob * right_count);
				if (cost < best_cost)
				{
					best_cost = cost;
					best_split_plane = split_plane;
				}
			}
		}
	}

	// Now that the best split plane has been found, determine if this cost is less than simply raytracing the node itself.
	if (best_cost < (intersect_cost * node->triangles.size ()))
	{
		left->box.min = node->box.min;
		left->box.max = node->box.max;
		left->box.max[split_axis] = best_split_plane;

		right->box.min = node->box.min;
		right->box.max = node->box.max;
		right->box.min[split_axis] = best_split_plane;

		left->box.calcDimensions ();
		right->box.calcDimensions ();

		// Insert the triangles into the left and right nodes.
		for (size_t i = 0; i < node->triangles.size (); ++i)
		{
			left->insertTriangle (node->triangles[i]);
			right->insertTriangle (node->triangles[i]);
		}

		// Clear the parent node, link it with the children, and finish.
		node->triangles.clear ();
		if (left->triangles.size () == 0)
		{
			delete left;
			left = NULL;
		}

		if (right->triangles.size () == 0)
		{
			delete right;
			right = NULL;
		}
		node->left = left;
		node->right = right;

		return true;
	}

	delete left;
	delete right;
	return false;
}

/**
 * Perform a mediam split.
 * @param node TriangleKDTreeNode to split.
 * @param current_depth depth in the tree.
 */
bool TriangleKDTree::medianSplit (TriangleKDTreeNode *&node, int depth)
{
	// Find the longest axis, this will be the splitting axis.
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

	float split_plane = (node->box.max[axis] + node->box.min[axis]) * (float)0.5;
	TriangleKDTreeNode *left = new TriangleKDTreeNode;
	TriangleKDTreeNode *right = new TriangleKDTreeNode;

	left->box.min = node->box.min;
	left->box.max = node->box.max;
	left->box.max[axis] = split_plane;

	right->box.min = node->box.min;
	right->box.max = node->box.max;
	right->box.min[axis] = split_plane;

	left->box.calcDimensions ();
	right->box.calcDimensions ();

	// Add the triangles to the new child nodes.
	for (size_t i = 0; i < node->triangles.size (); ++i)
	{
		bool insert_left = false;
		bool insert_right = false;
		insert_left = left->insertTriangle (node->triangles[i]);
		insert_right = right->insertTriangle (node->triangles[i]);
		/*if (!insert_left && !insert_right)
				{
					std::cout << "triangle missed: " << node->triangles[i]->m_vertices[0] << " " <<
								 node->triangles[i]->m_vertices[1] << " " <<
								 node->triangles[i]->m_vertices[2] << std::endl;
					std::cout << "\tleft, right center = " << left->center << ", " << right->center << std::endl;
					std::cout << "\tleft, right half size = " << left->half_size << ", " << right->half_size << std::endl;
				}*/
	}

	// Check to make sure that the split actually did something.
	if (node->triangles.size () == left->triangles.size () &&
			node->triangles.size () == right->triangles.size ())
	{
		delete left;
		delete right;
		return false;
	}

	if (left->triangles.size () == 0)
	{
		delete left;
		left = NULL;
	}

	if (right->triangles.size () == 0)
	{
		delete right;
		right = NULL;
	}

	node->triangles.clear ();
	node->left = left;
	node->right = right;
	return true;

}

/**
 * Render the borders of the tree.
 * @param node TriangleKDTreeNode to render currently.
 */
void TriangleKDTree::renderBordersR (const TriangleKDTreeNode *node) const
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

		renderBordersR (node->left);
		renderBordersR (node->right);
	}
#endif
}

/**
 * Render the contents of the tree.
 * @param node TriangleKDTreeNode to render.
 */
void TriangleKDTree::renderR (const TriangleKDTreeNode *node) const
{
#ifdef USE_GL
	if (node == NULL)
	{
		return;
	}

	renderR (node->left);
	renderR (node->right);

	glBegin (GL_TRIANGLES);

	for (size_t t = 0; t < node->triangles.size (); ++t)
	{
		for (int i = 0; i < 3; ++i)
		{
			math::vec3f normal = node->triangles[t]->m_normals[i];
			math::vec3f vertex = node->triangles[t]->m_vertices[i];
			glNormal3fv (normal.v);
			glVertex3fv (vertex.v);
		}
	}

	glEnd ();
#endif
}

/**
 * Recursively clear the tree.
 * @param node TriangleKDTreeNode to clear.
 */
void TriangleKDTree::clearR (TriangleKDTreeNode *&node)
{
	if (node == NULL)
	{
		return;
	}

	clearR (node->left);
	clearR (node->right);
	delete node;
	node = NULL;
}

/**
 * Split the tree.
 */
bool TriangleKDTree::split (TriangleKDTreeNode *&node, int depth)
{
	if (node->triangles.size () > 64)
	{
		return medianSplit (node, depth);
	}

	else if (node->triangles.size () > 32)
	{
		return sahSplit (node, depth);
	}

	return false;
}

const int profile_ray_id = -1;

/**
 * Recursively intersect with the tree.
 * @param node TriangleKDTreeNode to check.
 * @parm ray Ray to check against node.
 * @param tmin Minimum time value allowed for an intersection to occur.
 * @param info Intersection information to be returned.
 * @param shadow If this is a shadow ray, we can avoid some costly computations.
 */
void TriangleKDTree::intersectR (const TriangleKDTreeNode *node, Ray &ray, IntersectInfo &info)
{
	if (node == NULL) {
		return;
	}

#ifdef METRICS
	(const_cast<TriangleKDTreeNode*>(node))->numpointsvisited++;	
#endif
#ifdef TRACK_TRAVERSALS
	ray.num_nodes_traversed++;
	if (ray.id == profile_ray_id) {
		cout << node->id << endl;
		if (node->id == 4557) {
			int bb = 0;
		}
	}
#endif

#ifdef PROFILE
	m_profiler->record(0, node->triangles.size());
#endif
	float time = (float)0.0;
	if (node->checkRay (ray, time)) {
#ifdef PROFILE
		m_profiler->record(1, node->triangles.size());
#endif

		if (time > info.time) { // This box is farther away than the previously hit triangle, so we don't care about it.
			return;
		}
#ifdef PROFILE
		m_profiler->record(2, node->triangles.size());
#endif

		bool hit = false;
		if (node->isLeaf ()) {
#ifdef PROFILE
		m_profiler->record(3, node->triangles.size());
#endif

			time = (float)0.0;
			// Test the triangles in this node for an intersection.
			for (size_t i = 0; i < node->triangles.size (); ++i) {
				if (node->triangles[i]->checkRayIntersection (ray.origin, ray.direction, time)) {
#ifdef PROFILE
		m_profiler->record(4, 1);
#endif

					if (time >= 0.0f) {
#ifdef PROFILE
		m_profiler->record(5, 1);
#endif

						// Ensure that the hitpoint is actually within the bounds of this node.
						// The triangle could get hit outside of this node since triangles
						// can span multiple nodes.
						kvec hit_point = ray.origin + ray.direction * time;


						if (time < info.time) {
							hit = true;
#ifdef PROFILE
		m_profiler->record(6, 1);
#endif

							info.time		= time;
							info.hit_point 	= hit_point;
							info.normal		= node->triangles[i]->interpolateNormal (info.hit_point, info.barycentrics);
							info.material 	= node->triangles[i]->m_material;
							info.triangle	= node->triangles[i];
							if (info.material->texture.hasData ()) {
#ifdef PROFILE
		m_profiler->record(7, 1);
#endif

								info.tex_coord = node->triangles[i]->interpolateTexture (info.barycentrics);
							}
						}
					}
				}
			}
			if (hit) ray.intersects = true;
			return;
		}
	} else {
		return;
	}

	intersectR (node->left, ray, info);
	intersectR (node->right, ray, info);
}


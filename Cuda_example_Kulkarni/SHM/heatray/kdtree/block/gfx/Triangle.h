/*
   Filename : Triangle.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Class to define a triangle. 

   Change List:

      - 06/18/2009  - Created (Cody White)
 */

#pragma once

#include <math/Vector.h>
#include <math/Matrix.h>
#include <gfx/Material.h>
#include <vector>

namespace gfx
{

/**
 * Class to contain one triangle.
 */
class Triangle
{
public:


	Triangle (void);
	~Triangle (void);
	Triangle (const Triangle &rhs);
	Triangle & operator= (const Triangle &rhs);
	inline math::Vector <float, 3> & operator[] (int i)
	{
		return m_vertices[i];
	}

	math::Vector <float, 3> interpolateNormal (math::Vector <float, 3> hit_point, math::Vector <float, 3> &barycentrics);
	math::Vector <float, 2> interpolateTexture (math::Vector <float, 3> &barycentrics);
	bool checkRayIntersection (math::Vector <float, 3> &ray_origin, math::Vector <float, 3> &ray_direction, float &time);
	void calcTangents (void);
	void initializeIntersection (void);

	// Member variables
	math::Vector <float, 3> m_vertices [3];			// Vertices of this triangle.
	math::Vector <float, 2> m_texture_coords [3];	// Texture coordinates that correspond to the vertices of the triangle.
	math::Vector <float, 3> m_tangents [3];			// Tangent vectors at each vertex.
	math::Vector <float, 3> m_bitangents[3];		// Bitangent vectors at each vertex.
	math::Vector <float, 3> m_normals [3];			// Normals of this triangle at each vertex.
	gfx::Material    *m_material;			// Pointer to a material.

	bool m_has_split;	// Flag for determining if this triangle has tesselated data.

	std::vector <Triangle *> m_tesselations;	// The tesselated version of this triangle.

	// Intersection variables.
	math::Vector <float, 3> m_b, m_c, m_N;
	float m_barycentric_divide;
	float m_u, m_v;

private:

	void getTangent (math::Vector <float, 3> &v1, math::Vector <float, 3> &v2, math::Vector <float, 3> &v3, math::Vector <float, 2> &t1, math::Vector <float, 2> &t2, math::Vector <float, 2> &t3,
			math::Vector <float, 3> &tangent, math::Vector <float, 3> &bitangent);
};

}



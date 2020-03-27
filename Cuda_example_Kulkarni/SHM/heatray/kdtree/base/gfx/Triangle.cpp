/*
   Filename : Triangle.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Class to define a triangle.

   Change List:

      - 06/18/2009  - Created (Cody White)
 */

#include <gfx/Triangle.h>
using namespace gfx;

Triangle::Triangle (void)
{
	m_vertices[0].zero ();
	m_vertices[1].zero ();
	m_vertices[2].zero ();

	m_texture_coords[0].zero ();
	m_texture_coords[1].zero ();
	m_texture_coords[2].zero ();

	m_normals[0].zero ();
	m_normals[1].zero ();
	m_normals[2].zero ();

	m_tangents[0].zero ();
	m_tangents[1].zero ();
	m_tangents[2].zero ();

	m_has_split = false;
	m_material = NULL;
}

/**
 * Destructor.
 */
Triangle::~Triangle (void)
{
}

/**
 * Copy Constructor.
 * @param rhs Triangle to copy.
 */
Triangle::Triangle (const Triangle &rhs)
{
	*this = rhs;
}

/**
 * Operator =.
 * @param rhs Triangle to set this one equal to.
 */
Triangle & Triangle::operator= (const Triangle &rhs)
{
	if (this != &rhs)
	{
		m_vertices[0] = rhs.m_vertices[0];
		m_vertices[1] = rhs.m_vertices[1];
		m_vertices[2] = rhs.m_vertices[2];
		m_texture_coords[0] = rhs.m_texture_coords[0];
		m_texture_coords[1] = rhs.m_texture_coords[1];
		m_texture_coords[2] = rhs.m_texture_coords[2];
		m_tangents[0] = rhs.m_tangents[0];
		m_tangents[1] = rhs.m_tangents[1];
		m_tangents[2] = rhs.m_tangents[2];
		m_bitangents[0] = rhs.m_bitangents[0];
		m_bitangents[1] = rhs.m_bitangents[1];
		m_bitangents[2] = rhs.m_bitangents[2];
		m_normals[0]  = rhs.m_normals[0];
		m_normals[1]  = rhs.m_normals[1];
		m_normals[2]  = rhs.m_normals[2];
		m_material = rhs.m_material;
		m_has_split = rhs.m_has_split;
		m_b = rhs.m_b;
		m_c = rhs.m_c;
		m_N = rhs.m_N;
		m_barycentric_divide = rhs.m_barycentric_divide;
		m_u = rhs.m_u;
		m_v = rhs.m_v;
	}

	return *this;
}

/**
 * Calculate an interpolated normal for a hit point on this triangle.
 * @param hit_point The hit point on this triangle.
 */
math::Vector <float, 3> Triangle::interpolateNormal (math::Vector <float, 3> hit_point, math::Vector <float, 3> &barycentrics)
{
	math::Vector <float, 3> normal = math::normalize (math::cross (m_vertices[1] - m_vertices[0], m_vertices[2] - m_vertices[0]));
	float whole_area = math::dot (normal, math::cross (m_vertices[1] - m_vertices[0], m_vertices[2] - m_vertices[0]));

	// Compute first barycentric coordinate
	float area_p12 = math::dot (normal, math::cross (m_vertices[1] - hit_point, m_vertices[2] - hit_point));
	barycentrics[0] = area_p12 / whole_area;

	// Compute the second barycentric coordinate
	float area_p20 = math::dot (normal, math::cross (m_vertices[2] - hit_point, m_vertices[0] - hit_point));
	barycentrics[1] = area_p20 / whole_area;

	// Compute the last barycentric coordinate (a + b + c = 0)
	barycentrics[2] = (float)1 - barycentrics[0] - barycentrics[1];

	return math::normalize (m_normals[0] * barycentrics[0] + m_normals[1] * barycentrics[1] + m_normals[2] * barycentrics[2]);
}

/**
 * Calculate an interpolated texture coordinate for a hit point on this triangle.
 * We can use the barycentric coordinates already calculated from the normal interpolation code.
 * @param hit_point The hit point on this triangle.
 */
math::Vector <float, 2> Triangle::interpolateTexture (math::Vector <float, 3> &barycentrics)
{
	return (m_texture_coords[0] * barycentrics[0] + m_texture_coords[1] * barycentrics[1] + m_texture_coords[2] * barycentrics[2]);
}

/**
 * Check for a ray collision.
 * @param ray_origin Origin of the incoming ray.
 * @param ray_direction Direction of the incoming ray.
 * @param time Time of intersection.
 */
bool Triangle::checkRayIntersection (math::Vector <float, 3> &ray_origin, math::Vector <float, 3> &ray_direction, float &time)
{
	time = -math::dot (ray_origin - m_vertices[0], m_N) / math::dot (ray_direction, m_N);
	if (time < 0.0000001)
	{
		return false;
	}

	math::Vector <float, 3> H;
	H[m_u] = ray_origin[m_u] + time * ray_direction[m_u] - m_vertices[0][m_u];
	H[m_v] = ray_origin[m_v] + time * ray_direction[m_v] - m_vertices[0][m_v];

	float beta = (m_b[m_u] * H[m_v] - m_b[m_v] * H[m_u]) * m_barycentric_divide;
	if (beta < 0)
	{
		return false;
	}

	float gamma = (m_c[m_v] * H[m_u] - m_c[m_u] * H[m_v]) * m_barycentric_divide;
	if (gamma < 0)
	{
		return false;
	}

	if (beta + gamma > 1)
	{
		return false;
	}

	return true;
}

/**
 * Calculate the tangent at each of the vertices.  This is used for eye-space conversion.
 */
void Triangle::calcTangents (void)
{
	getTangent (m_vertices[0], m_vertices[1], m_vertices[2], m_texture_coords[0], m_texture_coords[1], m_texture_coords[2], m_tangents[0], m_bitangents[0]);
	getTangent (m_vertices[1], m_vertices[2], m_vertices[0], m_texture_coords[1], m_texture_coords[2], m_texture_coords[0], m_tangents[1], m_bitangents[1]);
	getTangent (m_vertices[2], m_vertices[0], m_vertices[1], m_texture_coords[2], m_texture_coords[0], m_texture_coords[1], m_tangents[2], m_bitangents[2]);
}

/**
 * Initialize intersection.
 */
void Triangle::initializeIntersection (void)
{
	m_b = m_vertices[2] - m_vertices[0];
	m_c = m_vertices[1] - m_vertices[0];
	m_N = math::cross (m_c, m_b);

	int k;
	if (fabs (m_N.x ()) > fabs (m_N.y ()))
	{
		if (fabs (m_N.x ()) > fabs (m_N.z ()))
		{
			k = 0;
		}
		else
		{
			k = 2;
		}
	}
	else
	{
		if (fabs (m_N.y ()) > fabs (m_N.z ()))
		{
			k = 1;
		}
		else
		{
			k = 2;
		}
	}

	int mod[] = {0, 1, 2, 0, 1};
	m_u = mod[k + 1];
	m_v = mod[k + 2];
	m_barycentric_divide = 1.0 / (m_b[m_u] * m_c[m_v] - m_b[m_v] * m_c[m_u]);
}


/**
 * Compute the tangent based on the passed in vertices and texture coordinates.
 * @param v1 Vertex 1 of the triangle.
 * @param v2 Vertex 2 of the triangle.
 * @param v3 Vertex 3 of the triangle.
 * @param t1 Texture coordinate 1 that corresponds to v1.
 * @param t2 Texture coordinate 2 that corresponds to v2.
 * @param t3 Texture coordinate 3 that corresponds to v3.
 */
void Triangle::getTangent (math::Vector <float, 3> &v1, math::Vector <float, 3> &v2, math::Vector <float, 3> &v3, math::Vector <float, 2> &t1, math::Vector <float, 2> &t2, math::Vector <float, 2> &t3,
		math::Vector <float, 3> &tangent, math::Vector <float, 3> &bitangent)
{
	math::Vector <float, 3> tan, bitan;

	math::Vector <float, 3> Q1 = v2 - v1;
	math::Vector <float, 3> Q2 = v3 - v1;
	math::Vector <float, 2> s = t2 - t1;
	math::Vector <float, 2> t = t3 - t1;

	float det = (float)1.0 / (s.x () * t.y () - s.y () * t.x ());
	tangent = ((Q1 * (t.y ())) - (Q2 * (s.y ()))) * det;
	bitangent = ((Q1 * (-t.x ())) + (Q2 * (s.x ()))) * det;
	tangent = normalize (tangent);
	bitangent = normalize (bitangent);

	math::Vector <float, 3> normal, b;
	normal = cross (tangent, bitangent);
	b = cross (normal, tangent);
	float handedness = dot (b, bitangent) < 0 ? 1 : -1;
	bitangent *= handedness;
}



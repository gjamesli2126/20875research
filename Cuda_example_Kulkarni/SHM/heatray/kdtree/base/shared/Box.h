/*

   Filename : Box.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Bounding box. 

   Change List:

      - 09/16/2010  - Created (Cody White)

*/

#pragma once

#include <math/Vector.h>
#include <gfx/Triangle.h>
#include <kdtree/TriangleBox.h>
#include <shared/Ray.h>

struct Box
{
	typedef math::Vector <float, 3> bvec;

	/**
	  * Default constructor.
	  */
	Box (void)
	{
		min 	= bvec::zero ();
		max 	= bvec::zero ();
		center 	= bvec::zero ();
	}

	/**
	  * Paramater constructor.
	  * @param _min Minimum value of the box.
	  * @param _max Maximum value of the box.
	  */
	Box (const bvec &_min, const bvec &_max)
	{
		min = _min;
		max = _max;
	}

	/**
	  * Operator=.
	  * @param other Box to set this box equal to.
	  */
	Box & operator= (const Box &other)
	{
		if (this != &other)
		{
			min 	= other.min;
			max 	= other.max;
			center 	= other.center;
		}

		return *this;
	}

	/**
	  * Get the area of the box.
	  */
	float getArea (void) const
	{
		float width  = (max[0] - min[0]);
		float length = (max[1] - min[1]);
		float height = (max[2] - min[2]);
		return (float)2.0 * (length * width   +
				         length * height +
						 width * height);
	}

	/**
	  * Test an intersection with a triangle.
	  * @param triangle Triangle to test against the box.
	  */
	bool testTriangle (const gfx::Triangle  *triangle) const
	{
		float tri_verts[3][3];
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				tri_verts[i][j] = triangle->m_vertices[i][j];
			}
		}

		return tribox::triBoxOverlap (center.v, half_size.v, tri_verts);
	}

	/**
	  * Test a point to see if it lies within this box.
	  * @param point Point to test.
	  */
	bool testPoint (const bvec &point) const
	{
		for (int i = 0; i < 3; ++i)
		{
			if (point[i] > max[i] || point[i] < min[i])
			{
				return false;
			}
		}

		return true;
	}

	/** 
	  * Test for a ray collision with this box.
	  * @param ray Ray to check with.
	  * @param time Distance from the ray to the box.
	  */
	bool testRay (const Ray  &ray, float &time) const
	{
		float min_time = (float)0.0, max_time = (float)HUGE_VAL;
		float epsilon = (float)0.0001;
		//printf("testing ray:origin=(%f %f %f) direction=(%f %f %f) inverse_direction=(%f %f %f) with box. min=(%f %f %f) max=(%f %f %f)\n",ray.origin[0],ray.origin[1],ray.origin[2],ray.direction[0],ray.direction[1],ray.direction[2], ray.inverse_direction[0],ray.inverse_direction[1],ray.inverse_direction[2],min[0],min[1],min[2],max[0], max[1],max[2]);

		// Close-in on the box in each dimension seperately.
		for (size_t d = 0; d < 3; ++d)
		{
			if (ray.direction[d] == (float)0.0)
			{
				if (ray.origin[d] > max[d] || ray.origin[d] < min[d])
				{
					return false;
				}
			}
			else
			{
				float time1 = (min[d] - epsilon - ray.origin[d]) * ray.inverse_direction[d];
				float time2 = (max[d] + epsilon - ray.origin[d]) * ray.inverse_direction[d];

				if (time1 > time2)
				{
					std::swap (time1, time2);
				}

				min_time = std::max (time1, min_time);
				max_time = std::min (time2, max_time);
			
				if ((time2 < min_time) || (time1 > max_time))
				{
					return false;
				}
			}
		}
		
		time = min_time;
		return true;
	}

	/**
	  * Determine the center and the half size of the box.
	  */
	void calcDimensions (void)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			center[i] = min[i] + (max[i] - min[i]) * (float)0.5;
			half_size[i] = fabs (max[i] - min[i]) * (float)0.5;
		}
	}

	/**
	  * Get the squared distance to a point from this box.
	  * @param point Point to compute distance from.
	  */
	float calcDistanceSquared (math::vec3f point) const
	{
		math::vec3f projected_point = min;
		for (size_t i = 0; i < 3; ++i)
		{
			if (point[i] > max[i])
			{
				projected_point[i] = max[i];
			}

			if (point[i] > min[i] && point[i] < max[i])
			{
				projected_point[i] = point[i];
			}
		}

		return math::length2 (projected_point - point);
	}

	/**
	  * Get the distance to a point from this box.
	  * @param point Point to compute distance from.
	  */
	float calcDistance (math::vec3f point) const
	{
		return sqrtf (calcDistanceSquared (point));
	}

	// Member variables.
	bvec min;		// Minimum value of this box.
	bvec max;   	// Maximum value of this box.
	bvec center;	// Center of the bounding box.
	bvec half_size;	// Half the size of the box.
};



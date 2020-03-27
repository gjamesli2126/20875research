/*

   Filename : IntersectInfo.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Struct to contain information about a ray-triangle intersection. 

   Change List:

      - 12/20/2009  - Created (Cody White)

*/

#pragma once

#include "math/Vector.h"
#include "gfx/Material.h"
#include "gfx/Triangle.h"

class IntersectInfo
{
public:
	IntersectInfo (void) {
		init();
	}

	void init() {
		time = HUGE_VAL;
		hit_point = math::Vector <float, 3>::zero ();
		normal 	  = math::Vector <float, 3>::zero ();
		tex_coord = math::Vector <float, 2>::zero ();
		material = NULL;
	}

	math::Vector <float, 3>	 hit_point;			// Point on the triangle intersected by the ray.
	math::Vector <float, 3>	 normal;			// Normal coordinate at the hit point.
	math::Vector <float, 2>  tex_coord;			// Texture coordinate at the hit point.
	gfx::Material   	 *material;			// Reference to the material for the triangle hit.
	float				 	 time;				// Time to the hit point along the ray.
	math::Vector <float, 3>	 barycentrics;		// Barycentric coordinates for this hit point.
	gfx::Triangle	 *triangle;			// Triangle hit by the ray.
};


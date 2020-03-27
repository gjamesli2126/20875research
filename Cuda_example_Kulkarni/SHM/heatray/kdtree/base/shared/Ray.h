/*

   Filename : Ray.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Ray definition file. 

   Change List:

      - 12/20/2009  - Created (Cody White)

*/

#pragma once

#include "math/Vector.h"
#include <stdint.h>

class Ray
{
public:
	math::Vector <float, 3> origin;				// Origin point of the ray.
	math::Vector <float, 3> direction;			// Normalized vector pointing in the direction of the ray.
	math::Vector <float, 3> inverse_direction; 	// Used for intersection tests.
	bool intersects;

	Ray() {
		init();
	}

	void init() {
#ifdef TRACK_TRAVERSALS
		num_nodes_traversed = 0;
		id = next_id++;
#endif
	}

#ifdef TRACK_TRAVERSALS
	uint64_t num_nodes_traversed;
	int id;
private:
	static int next_id;
#endif
};

/*

   Filename : Light.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Struct to contain a light source. 

   Change List:

      - 03/22/2011  - Created (Cody White)

*/

#pragma once

#include <math/Vector.h>
#include <fstream>

struct Light
{
	/**
	  * Default constructor.
	  */
	Light (void)
	{
		position = math::Vector <float, 3>::zero ();
		forward  = math::Vector <float, 3>::zero ();
		up		 = math::Vector <float, 3>::zero ();
		right	 = math::Vector <float, 3>::zero ();
		power 	 = math::Vector <float, 3>::zero ();

		fov_y = (float)0.0;
		fov_x = (float)0.0;
	}

	/**
	  * Copy constructor.
	  */
	Light (const Light &other)
	{
		position = other.position;
		forward = other.forward;
		up = other.up;
		right = other.right;
		power = other.power;
		fov_y = other.fov_y;
		fov_x = other.fov_x;
	}

	void output() {
		printf("light %f %f %f	%f %f %f	%f %f %f	%f %f %f	%f %f\n",
				position.v[0], position.v[1], position.v[2],
				forward.v[0], forward.v[1], forward.v[2],
				up.v[0], up.v[1], up.v[2],
				right.v[0], right.v[1], right.v[2],
				fov_y, fov_x);
	}

	void load(std::ifstream& fin, const math::Vector <float, 3>& p) {
		fin >> position.v[0] >> position.v[1] >> position.v[2];
		fin >> forward.v[0] >> forward.v[1] >> forward.v[2];
		fin >> up.v[0] >> up.v[1] >> up.v[2];
		fin >> right.v[0] >> right.v[1] >> right.v[2];
		fin >> fov_y;
		fin >> fov_x;
		power = p;
	}

	math::Vector <float, 3> position;	// Position of the light.
	math::Vector <float, 3> forward;	// Forward vector.
	math::Vector <float, 3> up;			// Up vector.
	math::Vector <float, 3> right;		// Right vector.
	math::Vector <float, 3> power;		// Power of the light.
	float 					fov_y;		// Field of view in the y in radians.
	float					fov_x;		// Field of view in the x in radians.
};


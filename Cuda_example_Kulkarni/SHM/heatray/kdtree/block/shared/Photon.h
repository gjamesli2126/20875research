/*

   Filename : Photon.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Struct to contain a photon. 

   Change List:

      - 03/22/2011  - Created (Cody White)

*/

#pragma once

#include <math/Vector.h>
#include "util/random.h"

struct Photon 
{
	/**
	  * Default constructor.
	  */
	Photon (void)
	{
		position = math::Vector <float, 3>::zero ();
		normal	 = math::Vector <float, 3>::zero ();
		power 	 = math::Vector <float, 3>::zero ();

		caustic = false;
		inside = false;
		index_of_refraction = (float)1.0; // Air.
	}

	/**
	  * Copy constructor.
	  * @param other Object to copy.
	  */
	Photon (const Photon &other)
	{
		*this = other;
	}

	/**
	  * Operator=.
	  * @param other Object to copy.
	  */
	Photon & operator= (const Photon &other)
	{
		if (this != &other)
		{
			position 	= other.position;
			normal 		= other.normal;
			power 		= other.power;
			direction	= other.direction;
			inside		= other.inside;
			caustic     = other.caustic;
			
			index_of_refraction = other.index_of_refraction;
			random = other.random;
		}

		return *this;
	}

	math::Vector <float, 3> position;	// World-space position of the photon.
	math::Vector <float, 3> direction;	// Direction of the photon.
	math::Vector <float, 3> normal;		// Normal vector at the hitpoint.
	math::Vector <float, 3> power;		// Power for this photon (RGB).
	
	bool caustic;				// Is this photon a caustic photon or not.
	bool inside;				// Photon is inside or outside of an object.
	float 	 index_of_refraction;	// Index of refraction for the current material the photon is inside.

	RandomFloat *random;
};


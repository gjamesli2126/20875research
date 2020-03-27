/*
   Filename : Material.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Definition of a material for a .obj file. 

   Change List:

      - 12/21/2009  - Created (Cody White)
*/

#pragma once

#include <math/Vector.h>
#include <gfx/Texture.h>
#include <util/string.h>

namespace gfx
{

struct Material
{
	typedef math::Vector <float, 3> mvec;	// Typdef for easy material vector.

	/**
	  * Default constructor.
	  */
	Material (void)
	{
		name 					= "";
		texture.texture_name 	= "";
		ambient					= mvec::zero ();
		diffuse 				= mvec::zero ();
		specular 				= mvec::zero ();
		transmissive 			= mvec::zero ();
		specular_exponent 		= (float)0.0;
		index_of_refraction 	= (float)0.0;
	}

	// Member variables.
	util::string name;		// Name of the material.
	mvec ambient;			// Ambient component.
	mvec diffuse;			// Diffuse component.
	mvec specular;			// Specular component.
	mvec transmissive;		// Transmissive component.
	float specular_exponent;	// Specular exponent.
	float index_of_refraction;	// Index of refraction for this material.
	gfx::Texture texture;	// Texture which is bound to this material.
};

}


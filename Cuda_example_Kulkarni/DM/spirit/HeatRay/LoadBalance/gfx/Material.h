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
#include<boost/serialization/string.hpp>

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
	
	Material& operator = (const Material& rhs)
	{
		if(this != &rhs)
		{
			//name = rhs.name.c_str();
			//name.clear();
			//name.assign(rhs.name.c_str());
			ambient = rhs.ambient;
			diffuse = rhs.diffuse;
			specular = rhs.specular;
			transmissive = rhs.transmissive;
			specular_exponent = rhs.specular_exponent;
			index_of_refraction = rhs.index_of_refraction;
			texture = rhs.texture;
		}
		return *this;			
	}	

	// Member variables.
	std::string name;		// Name of the material.
	mvec ambient;			// Ambient component.
	mvec diffuse;			// Diffuse component.
	mvec specular;			// Specular component.
	mvec transmissive;		// Transmissive component.
	float specular_exponent;	// Specular exponent.
	float index_of_refraction;	// Index of refraction for this material.
	gfx::Texture texture;	// Texture which is bound to this material.
	friend class boost::serialization::access;
	template<typename Archiver>
	void serialize(Archiver& ar, const unsigned int)
	{
		ar & ambient & diffuse & specular & transmissive & specular_exponent & index_of_refraction & texture & name;
	}

};

}


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
#include <boost/serialization/split_member.hpp>

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
		seedForRandom = 0;
		numInvocations = 0;
		random = NULL;
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
			seedForRandom = other.seedForRandom;
			numInvocations = other.numInvocations;
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
	long int seedForRandom;	//needed to let other process generate the same random no. is purely needed only in case of distributed setup.
	int numInvocations;		//needed to let other process generate the same random no. is purely needed only in case of distributed setup.
	
	RandomFloat *random;

	/*friend class boost::serialization::access;
	template<typename Archiver>
	void serialize(Archiver& ar, const unsigned int)
	{
		ar & position & direction & normal & power & caustic & inside & index_of_refraction & seedForRandom & numInvocations;
	}*/

	friend class boost::serialization::access;

	template<class Archive>
	void save(Archive& ar, const unsigned version) const 
	{
		ar & position & direction & normal & power & caustic & inside & index_of_refraction & seedForRandom & numInvocations;
	}

	template<class Archive>
	void load(Archive& ar, const unsigned version) 
	{
		if(random)
		{
			seedForRandom = random->getSeed();
			numInvocations = random->getNumInvocations();
		}
		ar & position & direction & normal & power & caustic & inside & index_of_refraction & seedForRandom & numInvocations;
	}

	BOOST_SERIALIZATION_SPLIT_MEMBER();

};


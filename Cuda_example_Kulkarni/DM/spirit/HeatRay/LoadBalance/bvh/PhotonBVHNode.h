/*

   Filename : PhotonBVHNode.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : BVH Node for photon mapping. 

   Change List:

      - 03/22/2011  - Created (Cody White)

*/

#pragma once

#include <shared/Photon.h>
#include <shared/Box.h>
#include <math/Vector.h>
#include <vector>

struct PhotonBVHNode
{
	/**
	  * Default constructor.
	  */
	PhotonBVHNode (void)
	{
		left = NULL;
		right = NULL;
	}

	// Member variables.
	PhotonBVHNode 		 	 *left;		// Left node.
	PhotonBVHNode 		 	 *right; 	// Right node.
	std::vector <Photon > photons;	// List of photons contained within this node.
	Box 			 	 box;		// Bounding box for this node.
};


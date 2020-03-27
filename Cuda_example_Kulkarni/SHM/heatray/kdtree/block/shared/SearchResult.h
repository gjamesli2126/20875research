/*

   Filename : SearchResult.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Contains structure for a result from knn in PhotonBVH. 

   Change List:

      - 03/29/2011  - Created (Cody White)

*/

#pragma once

#include <shared/Photon.h>

struct SearchResult
{
	SearchResult (void)
	{
		photon = NULL;
		distance = (float)0.0;
	}

	SearchResult (const SearchResult &other)
	{
		photon = other.photon;
		distance = other.distance;
	}

	Photon *photon;
	float distance;
};


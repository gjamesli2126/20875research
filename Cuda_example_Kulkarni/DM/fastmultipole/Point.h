/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __POINT_H
#define __POINT_H

#define DIMENSION 2
#define MAX_CHILDREN 4
#define NUM_POINTS_PER_CELL 8
#define MAX_LEVELS 64
#include <boost/serialization/serialization.hpp>
#include<limits.h>
#include<vector>
#include<float.h>

class Point {
    public:
#ifdef TRAVERSAL_PROFILE
	std::vector<int> visitedNodes;
	std::vector<int> nodeDepthAtTruncation;
#endif
    float coordX, coordY;
    long int nodesTraversed;
    int id;
    float mass;
    float potential;
    template<typename Archiver>
    void serialize(Archiver& ar, const unsigned int)
    {
#ifdef TRAVERSAL_PROFILE
	ar & visitedNodes & nodeDepthAtTruncation & nodesTraversed;
#else
	ar & nodesTraversed;
#endif
	ar & coordX & coordY & mass & potential & id;
    }

    Point(){nodesTraversed = 0;}
   };


typedef struct LocalData{
	friend class boost::serialization::access;
	int index;
	long int nodesTraversed;	
	template<typename Archiver>
	void serialize(Archiver& ar, const unsigned int) 
	{
	    ar & index & nodesTraversed ;
  	}

}LocalData;

typedef struct Box
{
	friend class boost::serialization::access;
	int startX, startY, endX, endY;
	Box(int sx, int ex, int sy, int ey):startX(sx), endX(ex), startY(sy), endY(ey) {}
	Box(){startX=INT_MAX;startY=INT_MAX;endX=-INT_MAX;endY=-INT_MAX;}
	template<typename Archiver>
	void serialize(Archiver& ar, const unsigned int) 
	{
	    ar & startX & startY & endX & endY;
  	}
}Box;

typedef std::vector<LocalData> TLocalDataVector; 
typedef std::vector<Point> TPointVector;
typedef std::pair<int, long int> TBlockId;

#endif

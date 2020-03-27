/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __POINT_H
#define __POINT_H

#define DIMENSION 7

#include<vector>
#include<float.h>
#include<math.h>

class Point {
    public:
#ifdef TRAVERSAL_PROFILE
	std::vector<int> visitedNodes;
	std::vector<int> nodeDepthAtTruncation;
#endif

    float pt[DIMENSION];
    long int nodesTraversed;
    float tau;
    long int corr;
    template<typename Archiver>
    void serialize(Archiver& ar, const unsigned int)
    {
#ifdef TRAVERSAL_PROFILE
	ar & visitedNodes & nodeDepthAtTruncation & nodesTraversed & tau & corr;
#else
	ar & nodesTraversed & tau & corr;
#endif

	for(int i=0;i<DIMENSION;i++)
		ar & pt[i];
    }

    Point(){tau = FLT_MAX; nodesTraversed = 0; corr=0;}
    Point& operator = (const Point& rhs)
    {
		nodesTraversed = rhs.nodesTraversed;
		tau = rhs.tau;
		corr = rhs.corr;
		for(int i=0;i<DIMENSION;i++)
			pt[i] = rhs.pt[i];
		return *this;
    }

   };

typedef std::vector<Point> TPointVector;
typedef std::pair<int, long int> TBlockId;

typedef struct LocalData{
	friend class boost::serialization::access;
	int index;
	long int nodesTraversed;	
	long int corr;
	template<typename Archiver>
	void serialize(Archiver& ar, const unsigned int) 
	{
	    ar & index & nodesTraversed & corr;
  	}

}LocalData;


typedef std::vector<LocalData> TLocalDataVector; 

class ComparePoints {
    public:
    int cord;
    ComparePoints(int  cord_) : cord(cord_ % DIMENSION) {};
    bool operator()(const Point& lhs, const Point& rhs) const {
	if((cord==0) && (lhs.pt[cord] == rhs.pt[cord]))
	{
		return lhs.pt[cord+1] < rhs.pt[cord+1];	
	}
	else	
        	return lhs.pt[cord] < rhs.pt[cord];
    }
};



#endif

/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __POINT_H
#define __POINT_H

#define DIMENSION 3
#include <boost/serialization/serialization.hpp>
#include<vector>
#include<float.h>

class Vec{
friend class boost::serialization::access;
public:
	float x; //represents the vector's magnitude in x axis.
	float y;//represents the vector's magnitude in y axis.
	float z;//represents the vector's magnitude in z axis.
	Vec() {x=0;y=0;z=0;}
	Vec(float f) { x = f; y = f; z = f; }
	template<typename Archiver>
    	void serialize(Archiver& ar, const unsigned int)
    	{
		ar & x & y & z;
    	}
	bool operator !=(const Vec& rhs) const
	{
		if((rhs.x != x) || (rhs.y != y) || (rhs.z != z))
			return false;
		else
			return true;
	}
	Vec& operator =(const Vec& rhs)
	{
		x = rhs.x;
		y = rhs.y;
		z = rhs.z;
		return *this;
	}
	bool operator <(const Vec& rhs) const
	{
		if(x < rhs.x)
			return true;
		else
		{
			if(x == rhs.x)
			{
				if(y < rhs.y)
					return true;
				else
				{
					if(y == rhs.y)
					{
						if(z < rhs.z)
							return true;
					}
				}
			}
		}
		return false;
	}
};

/*BH_Specific cell params */
class LeafParams{
public:
	Vec vel;
	Vec acc;
	LeafParams(Vec& vl, Vec& ac):vel(vl), acc(ac){}
};
class CellParams
{
	friend class boost::serialization::access;
public:
	Vec cofm;
	float mass;
	float ropen;
	CellParams():mass(0),ropen(0){cofm.x=cofm.y=cofm.z=0;}
	CellParams(float m, float rad, Vec& c):mass(m),ropen(rad)
	{
		cofm.x=c.x;cofm.y=c.y;cofm.z=c.z;	
	}
	template<typename Archiver>
    	void serialize(Archiver& ar, const unsigned int)
    	{
		ar & mass & ropen & cofm;
    	}
	CellParams& operator =(const CellParams& rhs)
	{
		mass = rhs.mass;
		ropen = rhs.ropen;
		cofm = rhs.cofm;
		return *this;
	}
	

};


class Point {
    public:
#ifdef TRAVERSAL_PROFILE
	std::vector<int> visitedNodes;
	std::vector<int> nodeDepthAtTruncation;
#endif
	float mass;
	Vec cofm; /* center of mass */
	Vec vel; /* current velocity */
	Vec acc; /* current acceleration */
    	long int nodesTraversed;
    	int id;
    template<typename Archiver>
    void serialize(Archiver& ar, const unsigned int)
    {
#ifdef TRAVERSAL_PROFILE
	ar & visitedNodes & nodeDepthAtTruncation & mass & cofm & vel & acc & nodesTraversed;
#else
	ar & mass & cofm & vel & acc & nodesTraversed;
#endif
    }

    Point(){nodesTraversed = 0;}
   };

typedef std::vector<Point> TPointVector;
typedef std::pair<int, long int> TBlockId;

typedef struct LocalData{
	friend class boost::serialization::access;
	int index;
	long int nodesTraversed;	
	template<typename Archiver>
	void serialize(Archiver& ar, const unsigned int) 
	{
	    ar & index & nodesTraversed;
  	}
}LocalData;


typedef std::vector<LocalData> TLocalDataVector; 

#endif

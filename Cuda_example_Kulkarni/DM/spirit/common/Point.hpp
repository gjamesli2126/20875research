/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef POINT_H
#define POINT_H

#define STRHELPER(x) #x
#define STR(x) STRHELPER(x)

#ifndef DIMENSION
#define DIMENSION 7
#endif


#pragma message "DIMENSION set to " STR(DIMENSION)

#include <boost/serialization/serialization.hpp>
#include<vector>
#include<float.h>
#include<math.h>
#include<stdio.h>

typedef enum TType{VPTREE, KDTREE, PCKDTREE, OCTREE}TType;

class Point {
    public:
    float pt[DIMENSION];
    long int nodesTraversed;
    long int id;
    Point(){nodesTraversed = 0; id=-1;}
    virtual ~Point(){}
    /*Point& operator = (const Point& rhs)
    {
		nodesTraversed = rhs.nodesTraversed;
		id = rhs.id;
		for(int i=0;i<DIMENSION;i++)
			pt[i] = rhs.pt[i];
		return *this;
    }*/
    template<typename Archiver>
    void serialize(Archiver& ar, const unsigned int){}

};

class VertexData
{
	public:
	VertexData* next;
	VertexData():next(NULL){}
	virtual ~VertexData() {}
};

typedef std::vector<Point*> TPointVector;
typedef std::pair<int, long int> TBlockId;

class Context{
public:
	long int index;
	Context(){}
	virtual ~Context(){}
	Context(long int i):index(i){}
};

typedef std::vector<Context*> TContextVector; 
enum status{TRUNC, LEFT, RIGHT};
class Vertex;
class SPIRITVisitor{
public:
	virtual status EvaluateVertex(Vertex* node, Point* p)=0;
	virtual void SetContext(Point* p, const Context* lv)=0;
	virtual Context* GetContext(const Point* p)=0;
	virtual void ProcessResult(Point* p)=0;
};


class InputFileParser
{
	public:
	virtual void ReadPoint(std::ifstream& input, Point* p)=0;
	virtual TType GetTreeType()=0;
};


#endif

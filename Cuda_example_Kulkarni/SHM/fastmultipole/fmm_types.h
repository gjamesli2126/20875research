/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __FMM_TYPES_H
#define __FMM_TYPES_H

#define DIMENSION 2
#define MAX_CHILDREN 4
#define NUM_POINTS_PER_CELL 1
#define MAX_LEVELS 64
#include<vector>
#include<float.h>
#include<limits.h>

typedef struct Vec {
double pt[DIMENSION];
Vec(double x){for(int i=0;i<DIMENSION;i++) pt[i]=x;}
}Vec;

typedef struct Point {
    double coordX, coordY;
    unsigned long long id;
    double mass;
    double potential;
    Point(const Point* p){coordX=p->coordX;coordY=p->coordY;id=p->id;mass=p->mass;potential=p->potential;}
    Point(){potential=0;}
}Point;

typedef struct Box
{
	double startX, startY, endX, endY;
	Box(double sx, double ex, double sy, double ey):startX(sx), endX(ex), startY(sy), endY(ey) {}
	Box(){startX=FLT_MAX;startY=FLT_MAX;endX=-FLT_MAX;endY=-FLT_MAX;}
}Box;

typedef struct VertexData{
	Point* p;
	VertexData* next;
	VertexData(Point* pt, bool clonePoint)
	{
		next=NULL;
		p=pt;
		if(clonePoint)
			p = new Point(pt);
	}
	VertexData(double mass)
	{
		p = new Point();
		p->mass=mass;	
		next = NULL;
	}
	~VertexData()
	{
		if(p)
			delete p;
	}
}VertexData;

typedef struct Vertex{
#if METRICS
	int footprint;
#endif
	Vertex* pChild[MAX_CHILDREN];
	Vertex* parent;
	std::vector<Vertex*> neighbors;
	long int label;
	short int level;
	bool isLeaf;
	Box box; 
	VertexData* vData;
	int numPointsInCell;
	Vertex():parent(0), level(0), label(0), isLeaf(false), numPointsInCell(0)
	{
		#ifdef METRICS
		footprint=0;
		#endif
		for(int i=0;i<MAX_CHILDREN;i++)
		{
			pChild[i]=0;
		}
	}
}Vertex;

typedef struct FuncIn{
std::vector<Vertex*>* allLeaves;
int stepNum;
int start;
int end;
}FuncIn;

typedef std::vector<Point> TPointVector;
typedef void (*thread_function)(void* funcIn, int start, int end, void* funcOut);
struct targs {
	thread_function func;
	int start, end;
	void *funcIn, *funcOut;
};


#endif

/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef OCTREE_TYPES_HPP
#define OCTREE_TYPES_HPP
#include"Point.hpp"

class Vec
{
	public:
	float pt[DIMENSION];
	Vec()
	{
		for(int i=0;i<DIMENSION;i++) pt[i]=0.0;
	}
	Vec(float x)
	{
		for(int i=0;i<DIMENSION;i++) pt[i]=x;
	}
	Vec(const Vec& rhs)
	{
		for(int i=0;i<DIMENSION;i++) pt[i]=rhs.pt[i];
	}
	Vec& operator =(const Vec& rhs)
	{
		for(int i=0;i<DIMENSION;i++) pt[i]=rhs.pt[i];
		return *this;
	}
	bool operator ==(const Vec& rhs)
	{
		bool flag = true;
		for(int i=0;i<DIMENSION;i++) 
		{
			if(pt[i] !=rhs.pt[i])
			{
				flag = false;
				break;
			}
		}
		return flag;
	}
	template<typename Archiver>
	void serialize(Archiver& ar, const unsigned int)
	{
		for(int i=0;i<DIMENSION;i++) ar & pt[i];
	}
};

class OctreePoint : public Point {
    public:
    float mass;
    Vec cofm, vel, acc;
    template<typename Archiver>
    void serialize(Archiver& ar, const unsigned int)
    {
	ar & nodesTraversed & id & mass & cofm & vel & acc;
	for(int i=0;i<DIMENSION;i++)
		ar & pt[i];
    }
    
    OctreePoint(){}
    OctreePoint(float m, const Vec& c, const Vec& v, const Vec& a)
    {
	mass = m; cofm = c; vel = v; acc = a;
	nodesTraversed = 0;
	id = -1;
    }
    ~OctreePoint(){}

    OctreePoint(const OctreePoint* p)
    {
	nodesTraversed = p->nodesTraversed; id=p->id;
	mass = p->mass; cofm = p->cofm; vel = p->vel; acc = p->acc;
	for(int i=0;i<DIMENSION;i++)
		pt[i] = p->pt[i];
    }

};


class OctreeVertexData : public VertexData
{
	public:
	float mass;
	Vec cofm;
	OctreePoint* p;
	OctreeVertexData():p(NULL){mass=0.0;}
	OctreeVertexData(OctreePoint* pt):p(pt){mass=0.0;}
	OctreeVertexData(float ma, Vec& co):p(NULL),mass(ma),cofm(co) {}
	OctreeVertexData(OctreePoint* pt, float ma, Vec& co, bool clonePoint):mass(ma),cofm(co)
	{
		if(clonePoint)
			p=new OctreePoint(pt);
		else
			p = pt;
	}
	~OctreeVertexData(){ if(p) delete p;}
	//void Initialize(OctreePoint* pt, float ma, Vec& co):p(pt){mass=ma;cofm=co;}
};


class OctreeContext: public Context{
public:
	long int nodesTraversed;	
	Vec acc;
	OctreeContext(){}
	~OctreeContext(){}
	OctreeContext(long int i, long int t, Vec& ac):Context(i),nodesTraversed(t),acc(ac){}
	OctreeContext(const OctreeContext* lc):Context(lc->index),nodesTraversed(lc->nodesTraversed),acc(lc->acc){}
    	template<typename Archiver>
    	void serialize(Archiver& ar, const unsigned int)
	{
		ar & nodesTraversed & index & acc;
	}
};

class OctreeInputFileParser: public InputFileParser
{
	TType type;
	public:
	OctreeInputFileParser(TType t):type(t){}
	void ReadPoint(std::ifstream& input, Point* p);
	TType GetTreeType(){return type;}
};




#endif

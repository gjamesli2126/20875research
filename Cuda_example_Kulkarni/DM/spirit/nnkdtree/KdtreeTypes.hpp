/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef KDTREE_TYPES_HPP
#define KDTREE_TYPES_HPP
#include"Point.hpp"


class KdtreePoint : public Point {
    public:
    float closest_dist;
    long int closest_label;
    template<typename Archiver>
    void serialize(Archiver& ar, const unsigned int)
    {
	ar & nodesTraversed & closest_dist & closest_label & id;
	for(int i=0;i<DIMENSION;i++)
		ar & pt[i];
    }

    KdtreePoint()
    {
      closest_dist = FLT_MAX; nodesTraversed = 0; closest_label=-1; id=-1;
    }
    ~KdtreePoint(){}
    /*KdtreePoint& operator = (const KdtreePoint& rhs)
    {
		nodesTraversed = rhs.nodesTraversed;
		tau = rhs.tau;
		id = rhs.id;
		closest_label = rhs.closest_label;
		for(int i=0;i<DIMENSION;i++)
			pt[i] = rhs.pt[i];
		return *this;
    }*/

    KdtreePoint(const KdtreePoint* p)
    {
	closest_dist = p->closest_dist; nodesTraversed = p->nodesTraversed; closest_label=p->closest_label; id=p->id;
	for(int i=0;i<DIMENSION;i++)
		pt[i] = p->pt[i];
    }

};

/*class KdtreeVertexData2 : public VertexData
{
	public:
	float min[DIMENSION], max[DIMENSION];
	float center[DIMENSION];
	float splitVal, boxDist;
	std::vector<float> splitVec;
	Point* p;
	KdtreeVertexData2():p(NULL){}
	KdtreeVertexData2(Point* pt):{p=new KdtreePoint(pt);}
	~KdtreeVertexData2(){if(p) delete p;}
};*/


class KdtreeVertexData : public VertexData
{
	public:
	float center[DIMENSION], splitVal, boxDist;
	Point* p;
	KdtreeVertexData():p(NULL){}
	KdtreeVertexData(Point* pt):p(pt){}
	KdtreeVertexData(float c[DIMENSION], float sp, float bd):p(NULL),splitVal(sp),boxDist(bd)
	{
		for(int i=0;i<DIMENSION;i++)
		{
			center[i] = c[i];
		}
	}
	~KdtreeVertexData(){if(p) delete p;}
	void Initialize(float c[DIMENSION], float sp, float bd)
	{
		for(int i=0;i<DIMENSION;i++)
		{
			center[i] = c[i];
		}
		splitVal = sp;
		boxDist = bd;
	}
};

class KdtreeContext: public Context{
public:
	long int nodesTraversed;	
	float closest_dist;
	long int closestLabel;
	KdtreeContext(){}
	~KdtreeContext(){}
	KdtreeContext(long int i, long int t, float ta, long int l):Context(i),nodesTraversed(t),closest_dist(ta),closestLabel(l){}
	KdtreeContext(const KdtreeContext* lc):Context(lc->index),nodesTraversed(lc->nodesTraversed),closest_dist(lc->closest_dist),closestLabel(lc->closestLabel){}
    	template<typename Archiver>
    	void serialize(Archiver& ar, const unsigned int)
	{
		ar & nodesTraversed & closest_dist & closestLabel & index;
	}
};

class KDTreeInputFileParser: public InputFileParser
{
	TType type;
	public:
	KDTreeInputFileParser(TType t):type(t){}
	void ReadPoint(std::ifstream& input, Point* p);
	TType GetTreeType(){return type;}
};

#endif

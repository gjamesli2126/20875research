/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef VPTREE_TYPES_HPP
#define VPTREE_TYPES_HPP
#include"Point.hpp"


class VptreePoint : public Point {
    public:
    float tau;
    long int closest_label;
    template<typename Archiver>
    void serialize(Archiver& ar, const unsigned int)
    {
	ar & nodesTraversed & tau & closest_label & id;
	for(int i=0;i<DIMENSION;i++)
		ar & pt[i];
    }

    VptreePoint()
    {
      tau = FLT_MAX; nodesTraversed = 0; closest_label=-1; id=-1;
    }
    ~VptreePoint(){}
    /*VptreePoint& operator = (const VptreePoint& rhs)
    {
		nodesTraversed = rhs.nodesTraversed;
		tau = rhs.tau;
		id = rhs.id;
		closest_label = rhs.closest_label;
		for(int i=0;i<DIMENSION;i++)
			pt[i] = rhs.pt[i];
		return *this;
    }*/

    VptreePoint(const VptreePoint* p)
    {
	tau = p->tau; nodesTraversed = p->nodesTraversed; closest_label=p->closest_label; id=p->id;
	for(int i=0;i<DIMENSION;i++)
		pt[i] = p->pt[i];
    }

};

class VptreeVertexData : public VertexData
{
	public:
	bool isLeftChild;
	float threshold;
	VptreePoint* p;
	VptreePoint* parent;
	float parentThreshold;
	VptreeVertexData():p(NULL),parent(NULL){}
	VptreeVertexData(float t, VptreePoint* pt):threshold(t),p(pt),parent(NULL){}
    	VptreeVertexData(const VptreeVertexData* parent):parentThreshold(parent->threshold),parent(new VptreePoint(parent->p)),p(NULL){}
	~VptreeVertexData(){if(p) delete p;}
	void Initialize(float t, VptreePoint* pt){threshold=t;p=pt;}
	template<class Archive>
	void save(Archive& ar, const unsigned version) const 
	{
		ar & threshold & *p;
	}
	
	template<class Archive>
	void load(Archive& ar, const unsigned version) 
	{
		VptreePoint pt;
		ar & threshold & pt;
		p = new VptreePoint(&pt);
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER();
};

class VptreeContext: public Context{
public:
	long int nodesTraversed;	
	float tau;
	long int closestLabel;
	VptreeContext(){}
	~VptreeContext(){}
	VptreeContext(long int i, long int t, float ta, long int l):Context(i),nodesTraversed(t),tau(ta),closestLabel(l){}
	VptreeContext(const VptreeContext* lc):Context(lc->index),nodesTraversed(lc->nodesTraversed),tau(lc->tau),closestLabel(lc->closestLabel){}
    	template<typename Archiver>
    	void serialize(Archiver& ar, const unsigned int)
	{
		ar & nodesTraversed & tau & closestLabel & index;
	}
};

class VPTreeInputFileParser: public InputFileParser
{
	TType type;
	public:
	VPTreeInputFileParser(TType t):type(t){}
	void ReadPoint(std::ifstream& input, Point* p);
	TType GetTreeType();
};

#endif

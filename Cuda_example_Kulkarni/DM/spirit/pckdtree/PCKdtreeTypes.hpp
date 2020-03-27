/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef PCKDTREE_TYPES_HPP
#define PCKDTREE_TYPES_HPP
#include"Point.hpp"


class PCKdtreePoint : public Point {
    public:
    long int corr;
    template<typename Archiver>
    void serialize(Archiver& ar, const unsigned int)
    {
	ar & nodesTraversed & corr & id;
	for(int i=0;i<DIMENSION;i++)
		ar & pt[i];
    }

    PCKdtreePoint()
    {
      corr = 0; nodesTraversed = 0; id=-1;
    }
    ~PCKdtreePoint(){}
    /*PCKdtreePoint& operator = (const PCKdtreePoint& rhs)
    {
		nodesTraversed = rhs.nodesTraversed;
		tau = rhs.tau;
		id = rhs.id;
		closest_label = rhs.closest_label;
		for(int i=0;i<DIMENSION;i++)
			pt[i] = rhs.pt[i];
		return *this;
    }*/

    PCKdtreePoint(const PCKdtreePoint* p)
    {
	corr = p->corr; nodesTraversed = p->nodesTraversed; id=p->id;
	for(int i=0;i<DIMENSION;i++)
		pt[i] = p->pt[i];
    }

};

class PCKdtreeContext: public Context{
public:
	long int nodesTraversed;	
	long int corr;
	PCKdtreeContext(){}
	~PCKdtreeContext(){}
	PCKdtreeContext(long int i, long int t, long int co):Context(i),nodesTraversed(t),corr(co){}
	PCKdtreeContext(const PCKdtreeContext* lc):Context(lc->index),nodesTraversed(lc->nodesTraversed),corr(lc->corr){}
    	template<typename Archiver>
    	void serialize(Archiver& ar, const unsigned int)
	{
		ar & nodesTraversed & corr & index;
	}
};

class PCKDTreeInputFileParser: public InputFileParser
{
	TType type;
	public:
	PCKDTreeInputFileParser(TType t):type(t){}
	void ReadPoint(std::ifstream& input, Point* p);
	TType GetTreeType(){return type;}
};

#endif

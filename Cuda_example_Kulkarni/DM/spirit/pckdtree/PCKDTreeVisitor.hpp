/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef PCKDTREEVISITOR_HPP
#define PCKDTREEVISITOR_HPP
#include "Point.hpp"
class PCKDTreeVisitor : public SPIRITVisitor
{
	double radius;
	long int corr;
public:
	long int GetCorr(){return corr;}
	PCKDTreeVisitor(double rad):radius(rad),corr(0){}
	status EvaluateVertex(Vertex* v, Point* p);
	void SetContext(Point* p, const Context* l);
	Context* GetContext(const Point* p);
	void ProcessResult(Point* pt);
};

#endif

/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef VPTREEVISITOR_HPP
#define VPTREEVISITOR_HPP
#include "Point.hpp"
class VPTreeVisitor : public SPIRITVisitor
{
	float Distance(const Point& a, const Point& b); 
public:
	VPTreeVisitor(){}
	status EvaluateVertex(Vertex* v, Point* p);
	void SetContext(Point* p, const Context* l);
	Context* GetContext(const Point* p);
	void ProcessResult(Point* p);
};

#endif

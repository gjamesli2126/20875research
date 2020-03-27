/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef KDTREEVISITOR_HPP
#define KDTREEVISITOR_HPP
#include "Point.hpp"
class KDTreeVisitor : public SPIRITVisitor
{
public:
	KDTreeVisitor(){}
	status EvaluateVertex(Vertex* v, Point* p);
	void SetContext(Point* p, const Context* l);
	Context* GetContext(const Point* p);
	void ProcessResult(Point* p);
};

#endif

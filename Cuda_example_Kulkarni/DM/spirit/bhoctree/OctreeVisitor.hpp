/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef OCTREEVISITOR_HPP
#define OCTREEVISITOR_HPP
#include "Point.hpp"
//extern long int pointsVisited;
class Vec;
class OctreeVisitor : public SPIRITVisitor
{
	float dtime, eps, tol, dia;
	void InitializeParams();
	void UpdatePointAcc(Point* pot, float mass, Vec& dr, float drsq, float epssq);
	bool AreSamePoints(const Point* pot, const Point* vData);
public:
	OctreeVisitor(float dt, float e, float t, float di):dtime(dt), eps(e), tol(t), dia(di){InitializeParams();}
	status EvaluateVertex(Vertex* v, Point* p);
	void SetContext(Point* p, const Context* l);
	Context* GetContext(const Point* p);
	void ProcessResult(Point* p);
};

#endif

/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "VPTreeVisitor.hpp"
#include "Point.hpp"
#include "SPIRIT.hpp"
#include<fstream>
status VPTreeVisitor::EvaluateVertex(Vertex* node, Point* pt)
{
	VptreePoint* p = (VptreePoint*)(pt);
	p->nodesTraversed++;
	VptreeVertexData* vData = (VptreeVertexData*)node->vData;
	if(vData->parent)
	{
		float upperDist = Distance(*(vData->parent), *p);
		if (!vData->isLeftChild) 
		{
			if ((upperDist + p->tau) < vData->parentThreshold) 
			{
				return TRUNC;
			}
		} 
		else
		{
			if ((upperDist - p->tau) > vData->parentThreshold)
			{	
				return TRUNC;
			}
		}
	}
	
	float dist=0.0;
	while(vData)
	{
		dist = Distance(*(vData->p), *p);
		if (dist < p->tau) 
		{
			p->closest_label = node->label;
			p->tau = dist;
		}
		vData = (VptreeVertexData*)(vData->next);
	}
	//for leaf nodes, return status does not make a difference as it is handled in WLManager's VisitNode function.
	vData = (VptreeVertexData*)node->vData;
	if(dist < vData->threshold)
	{
		return LEFT;
	}
	else
	{
		return RIGHT;
	}
}

Context* VPTreeVisitor::GetContext(const Point* pt)
{
	const VptreePoint* p = (const VptreePoint*)(pt);
	VptreeContext* tmpl = new VptreeContext();
	tmpl->nodesTraversed = p->nodesTraversed;
	tmpl->tau = p->tau;
	tmpl->closestLabel = p->closest_label;
	return tmpl;
}


void VPTreeVisitor::SetContext(Point* pt, const Context* l)
{
	VptreePoint* p = (VptreePoint*)(pt);
	const VptreeContext* tmpl = reinterpret_cast<const VptreeContext*>(l);
	p->nodesTraversed = tmpl->nodesTraversed;
	if(p->tau > tmpl->tau)
	{
		p->tau = tmpl->tau; 
		p->closest_label = tmpl->closestLabel; 
	}
}

void VPTreeVisitor::ProcessResult(Point* p)
{
}

float VPTreeVisitor::Distance(const Point& a, const Point& b) 
{
	float d = 0;
	for(int i = 0; i < DIMENSION; i++) {
		float diff = a.pt[i] - b.pt[i];
		d += diff * diff;
	}
	return sqrt(d);
}

void VPTreeInputFileParser::ReadPoint(std::ifstream& input, Point* p)
{
	VptreePoint* vpp=(VptreePoint*)(p);
	
	if(vpp)
	{
		for(int i=0;i<DIMENSION;i++)
			input >> (vpp->pt)[i];
		vpp->tau = FLT_MAX;
		vpp->nodesTraversed = 0;
		vpp->closest_label = -1;
	}
	return;
}

TType VPTreeInputFileParser::GetTreeType()
{
	return type;
}

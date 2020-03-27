/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "KDTreeVisitor.hpp"
#include "SPIRIT.hpp"
#include<fstream>

status KDTreeVisitor::EvaluateVertex(Vertex* node, Point* pt)
{
	pt->nodesTraversed++;
	KdtreeVertexData* vData = (KdtreeVertexData*)(node->vData);
	KdtreePoint* p = (KdtreePoint*)(pt);
	if((node->leftChild == NULL) && (node->rightChild == NULL))
	{
		while(vData)
		{
			float distSq=0;
			for(int i=0;i<DIMENSION;i++)
			{
				float tmp = (p->pt[i] - vData->p->pt[i]);
				distSq+= (tmp * tmp);	
			}
			
			float dist = sqrt(distSq);
			if (dist < p->closest_dist)
			{
				p->closest_dist = dist;
				p->closest_label = node->label;
			}
			vData = (KdtreeVertexData*)(vData->next);
		}
		return TRUNC;
	}

	float sumSq=0;
	for (int i = 0 ; i < DIMENSION; i++) 
	{
		float dist = p->pt[i] - vData->center[i];
		sumSq = sumSq + (dist * dist);
	}
	float sum = sqrt(sumSq);	
	if ((sum - vData->boxDist) < p->closest_dist) 
	{
		int axis = node->level % DIMENSION;
		if(p->pt[axis] < vData->splitVal)
		{
			return LEFT;
		}
		else
		{
			return RIGHT;
		}
	}
	return TRUNC;
}

Context* KDTreeVisitor::GetContext(const Point* pt)
{
	const KdtreePoint* p = (const KdtreePoint*)(pt);
	KdtreeContext* tmpl = new KdtreeContext();
	tmpl->nodesTraversed = p->nodesTraversed;
	tmpl->closest_dist = p->closest_dist;
	tmpl->closestLabel = p->closest_label;
	return tmpl;
}


void KDTreeVisitor::SetContext(Point* pt, const Context* l)
{
	KdtreePoint* p = (KdtreePoint*)(pt);
	const KdtreeContext* tmpl = reinterpret_cast<const KdtreeContext*>(l);
	p->nodesTraversed = tmpl->nodesTraversed;
	if(p->closest_dist > tmpl->closest_dist)
	{
		p->closest_dist = tmpl->closest_dist; 
		p->closest_label = tmpl->closestLabel; 
	}
}


void KDTreeVisitor::ProcessResult(Point* p)
{
}

void KDTreeInputFileParser::ReadPoint(std::ifstream& input, Point* p)
{
	KdtreePoint* kdp=(KdtreePoint*)(p);
	
	if(kdp)
	{
		for(int i=0;i<DIMENSION;i++)
			input >> (kdp->pt)[i];
		kdp->closest_dist = FLT_MAX;
		kdp->nodesTraversed = 0;
		kdp->closest_label = -1;
	}
	return;
}

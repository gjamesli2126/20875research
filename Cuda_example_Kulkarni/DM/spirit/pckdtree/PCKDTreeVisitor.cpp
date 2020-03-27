/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "PCKDTreeVisitor.hpp"
#include "SPIRIT.hpp"
#include<fstream>

status PCKDTreeVisitor::EvaluateVertex(Vertex* node, Point* pt)
{
	pt->nodesTraversed++;
	KdtreeVertexData* vData = (KdtreeVertexData*)(node->vData);
	PCKdtreePoint* p = (PCKdtreePoint*)(pt);
	if((node->leftChild == NULL) && (node->rightChild == NULL))
	{
		while(vData)
		{
			double distSq=0;
			for(int i=0;i<DIMENSION;i++)
			{
				double tmp = (p->pt[i] - vData->p->pt[i]);
				distSq+= (tmp * tmp);	
			}
			
			double dist = sqrt(distSq);
			if (dist < radius)
			{
				p->corr++;
			}
			vData = (KdtreeVertexData*)(vData->next);
		}
		return TRUNC;
	}

	double sumSq=0;
	for (int i = 0 ; i < DIMENSION; i++) 
	{
		double dist = p->pt[i] - vData->center[i];
		sumSq = sumSq + (dist * dist);
	}
	double sum = sqrt(sumSq);	
	if ((sum - vData->boxDist) < radius) 
	{
		return LEFT;
	}
	return TRUNC;
}

Context* PCKDTreeVisitor::GetContext(const Point* pt)
{
	const PCKdtreePoint* p = (const PCKdtreePoint*)(pt);
	PCKdtreeContext* tmpl = new PCKdtreeContext();
	tmpl->nodesTraversed = p->nodesTraversed;
	tmpl->corr = p->corr;
	return tmpl;
}


void PCKDTreeVisitor::SetContext(Point* pt, const Context* l)
{
	PCKdtreePoint* p = (PCKdtreePoint*)(pt);
	const PCKdtreeContext* tmpl = reinterpret_cast<const PCKdtreeContext*>(l);
	p->nodesTraversed = tmpl->nodesTraversed;
	p->corr = tmpl->corr; 
}


void PCKDTreeVisitor::ProcessResult(Point* pt)
{
	PCKdtreePoint* p = (PCKdtreePoint*)(pt);
	corr += p->corr;
	//printf("%d nodesTraversed %d\n",p->id, p->nodesTraversed);
}

void PCKDTreeInputFileParser::ReadPoint(std::ifstream& input, Point* p)
{
	PCKdtreePoint* kdp=NULL;
	kdp = (PCKdtreePoint*)(p);
	if(kdp)
	{
		for(int i=0;i<DIMENSION;i++)
			input >> (kdp->pt)[i];
		kdp->nodesTraversed = 0;
		kdp->corr = 0;
	}
	return;
}


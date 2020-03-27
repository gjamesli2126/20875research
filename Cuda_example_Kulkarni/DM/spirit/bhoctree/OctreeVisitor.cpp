/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "OctreeVisitor.hpp"
#include "SPIRIT.hpp"
#include<fstream>

void OctreeVisitor::InitializeParams()
{
	dtime = dtime * 0.5;
	eps = eps * eps;
	tol = 1.0/(tol*tol);
	dia = (dia * dia * tol);
}

void OctreeVisitor::UpdatePointAcc(Point* pot, float mass, Vec& dr, float drsq, float epssq)
{
	OctreePoint* p = (OctreePoint*)(pot);
	drsq+=epssq;
	float idr=(1.0/sqrt(drsq));
	float nphi=(mass*idr);
	float scale=((nphi*idr)*idr);
	for(int i=0;i<DIMENSION;i++)
		p->acc.pt[i] += dr.pt[i] * scale;
}

bool OctreeVisitor::AreSamePoints(const Point* pot, const Point* vData)
{
	bool ret  = true;
	if(pot->id != vData->id)
		ret = false;
	return ret;
}

status OctreeVisitor::EvaluateVertex(Vertex* node, Point* pt)
{
	//pointsVisited +=1;
	pt->nodesTraversed++;
	OctreeVertexData* vData = (OctreeVertexData*)(node->vData);
	OctreePoint* p = (OctreePoint*)(pt);
	float tmpdsq;
	if(node->level >0)
		tmpdsq = dia * ((float)1.0f)/((float)(1<<(2*node->level)));
	else
		tmpdsq = dia;

	Vec dr;
	float drsq=0.0;
	for(int i=0;i<DIMENSION;i++)
	{
		dr.pt[i]= (vData->cofm.pt[i] - p->cofm.pt[i]);
		drsq += dr.pt[i] * dr.pt[i];
	}
		
	if(drsq >= tmpdsq)
	{
		drsq += eps;
		UpdatePointAcc(pt, vData->mass, dr, drsq, eps);
		return TRUNC;
	}
	else
	{
		if(node->leaf) 
		{
			OctreeVertexData* tmpVData = vData;
			while(tmpVData)
			{
				if(!AreSamePoints(pt,tmpVData->p))
					UpdatePointAcc(pt, tmpVData->p->mass, dr, drsq, eps);
				tmpVData = (OctreeVertexData*)(tmpVData->next);
			}
			return TRUNC;
		}
	}
	return LEFT;
}

Context* OctreeVisitor::GetContext(const Point* pt)
{
	const OctreePoint* p = (const OctreePoint*)(pt);
	OctreeContext* tmpl = new OctreeContext();
	tmpl->nodesTraversed = p->nodesTraversed;
	tmpl->acc = p->acc;
	return tmpl;
}


void OctreeVisitor::SetContext(Point* pt, const Context* l)
{
	OctreePoint* p = (OctreePoint*)(pt);
	const OctreeContext* tmpl = reinterpret_cast<const OctreeContext*>(l);
	p->nodesTraversed = tmpl->nodesTraversed;
	p->acc = tmpl->acc; 
}

void OctreeVisitor::ProcessResult(Point* p)
{
	/*if(p->id == 6)
	{
		OctreePoint* op = reinterpret_cast<OctreePoint*>(p);
		printf("%d cofm:( %f %f %f ) acc:( %f %f %f ) nodesTraversed: %ld\n",op->id,op->cofm.pt[0],op->cofm.pt[1], op->cofm.pt[2], op->acc.pt[0],op->acc.pt[1], op->acc.pt[2],op->nodesTraversed);
	}*/
}

void OctreeInputFileParser::ReadPoint(std::ifstream& input, Point* p)
{
	OctreePoint* ocp = NULL;
	ocp=(OctreePoint*)(p);

	float params;
	if(ocp)
	{
		input >> ocp->mass; 
		for(int i=0;i<DIMENSION;i++)
		{
			input >> ocp->cofm.pt[i];
		}
		for(int i=0;i<DIMENSION;i++)
		{
			input >> ocp->vel.pt[i];
		}
		//printf("%f %f %f %f %f %f %f\n",ocp->mass,ocp->cofm.pt[0],ocp->cofm.pt[1],ocp->cofm.pt[2],ocp->vel.pt[0],ocp->vel.pt[1],ocp->vel.pt[2]);
		ocp->nodesTraversed = 0;
	}
	else
	{
		input >>params;
		//printf("%f\n",params);
	}

	return;
}


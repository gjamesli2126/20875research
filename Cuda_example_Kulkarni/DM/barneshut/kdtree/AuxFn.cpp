/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include"AuxFn.h"
void ReadPoints(std::ifstream& input, long int startIndex, long int endIndex, Point* points)
{
	long int numPoints = -1, tmpIndex=-1;

        while(true) 
	{
	    numPoints++;
	    float mass, cofmx, cofmy, cofmz, velx, vely, velz;
            input >> mass >> cofmx >> cofmy >> cofmz >> velx >> vely >>velz;
            
	    if(numPoints < startIndex)
		continue;

	    if(numPoints == endIndex)
		break;

	    tmpIndex++;
	    Point* p=&(points[tmpIndex]);
	    p->cofm.x = cofmx;
	    p->cofm.y = cofmy;
	    p->cofm.z = cofmz;
	    p->vel.x = velx;
	    p->vel.y = vely;
	    p->vel.z = velz;
	    p->acc.x = 0;
	    p->acc.y = 0;
	    p->acc.z = 0;
	    p->mass = mass;
	    p->nodesTraversed = 0;
	    p->id=numPoints+1;
        }
}

bool ReadPointsInBatch(std::ifstream& input, Point* points, long int totalNumPoints, int& batchSize)
{
	static long int numPoints = 0;
	long int tmpIndex=0;

        while(true) {
	    numPoints++;
	    tmpIndex++;
	    
	    float mass, cofmx, cofmy, cofmz, velx, vely, velz;
            input >> mass >> cofmx >> cofmy >> cofmz >> velx >> vely >>velz;

	    Point* p = &(points[tmpIndex-1]);
	    p->cofm.x = cofmx;
	    p->cofm.y = cofmy;
	    p->cofm.z = cofmz;
	    p->vel.x = velx;
	    p->vel.y = vely;
	    p->vel.z = velz;
	    p->acc.x = 0;
	    p->acc.y = 0;
	    p->acc.z = 0;
	    p->mass = mass;
	    p->nodesTraversed = 0;
	    p->id=numPoints;
	    if((numPoints == totalNumPoints)||(tmpIndex==batchSize))
			break;
        }
	batchSize = tmpIndex;
	if(numPoints != totalNumPoints)
		return true;
	else
		return false;
}

void WritePoints(std::ofstream& output, int procRank, void* inputSet, long int start, long int end)
{
	Point* tmpArr;
	TPointVector ptv;
	if(procRank == 0)
	{
		tmpArr = (Point *)inputSet;
		for(long int i=start;i<=end;i++)
		{
			int j=0;
			output <<tmpArr[i].cofm.x;
			output <<tmpArr[i].cofm.y;
			output <<tmpArr[i].cofm.z;
			output<<std::endl;
		}
	}
	else
	{
		ptv = *((TPointVector*)inputSet);
		for(long int i=start;i<=end;i++)
		{
			output <<ptv[i].cofm.x;
			output <<ptv[i].cofm.y;
			output <<ptv[i].cofm.z;
			output<<std::endl;
		}
	}
}

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
	    float coordX, coordY;
		input >> coordX;
		input >> coordY;

	    if(numPoints < startIndex)
		continue;

	    if(numPoints == endIndex)
		break;
	    tmpIndex++;
	    Point* p=&(points[tmpIndex]);
		p->coordX = coordX;
		p->coordY = coordY;
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
	    float coordX, coordY;
		input >> coordX;
		input >> coordY;
	    
	    Point* p = &(points[tmpIndex-1]);
		p->coordX = coordX;
		p->coordY = coordY;
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
			output <<tmpArr[i].coordX <<" "<<tmpArr[i].coordY;
			output<<std::endl;
		}
	}
	else
	{
		ptv = *((TPointVector*)inputSet);
		for(long int i=start;i<=end;i++)
		{
			output <<ptv[i].coordX <<" "<<ptv[i].coordY;
			output<<std::endl;
		}
	}
}

#ifdef SPAD_2
void ReadBottleneckDetails(std::ifstream& input, int numBottlenecks, std::vector<BneckSubtreeHdr>& bottlenecks)
{
        while(true) 
	{
		int procID;
		int childNum;
		long int pLeafLabel;
		input >> procID >> pLeafLabel >> childNum;
		BneckSubtreeHdr hdr(procID, childNum, pLeafLabel);
		bottlenecks.push_back(hdr);
		if(bottlenecks.size() == numBottlenecks)
			break;
        }
}
#endif

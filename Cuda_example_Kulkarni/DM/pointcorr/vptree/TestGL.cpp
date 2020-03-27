/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include<string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stack>
#include <queue>
#include <iomanip>
#include <time.h>
#include<math.h>
#include "GAL.h"
#include "CorrelationVisitor.h"
#include <boost/mpi.hpp>
#ifdef PAPI
#include"papi.h"
void handle_error(int retval)
{
	printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	exit(1);
}
#endif


class CommLineOpts
{
private:	
	typedef std::vector<std::pair<std::string, std::string> > clopts;
	clopts paramList;
	void ParseCommLineOpts(int argc, char **argv, char c='=');
public:
	CommLineOpts(int argc, char **argv)
	{
		ParseCommLineOpts(argc,argv,'=');
	}

	//bool Get(const std::string& paramName, int& val);
	bool Get(const std::string& paramName, std::vector<int>& val);
	void ParseCommaSeperatedValues(const char *str, std::vector<int>& retVal);
};


TPointVector points;

TPointVector& ReadPoints(const char *pInputFile)
{
    std::ifstream input(pInputFile, std::fstream::in);
	if(input.fail())
	{
		std::cout<<"File does not exist. exiting"<<std::endl;
		MPI_Finalize();
		exit(0);
	}
    
    if ( input.peek() != EOF ) {
        while(true) {
		Point t ;
		float coord[DIMENSION];
		for(int i=0;i<DIMENSION;i++)
		{
			input >> coord[i];
			t.pt[i] = coord[i];
		}
	    if(input.eof())
			break;
	    //printf("point read (%f %f)\n",t.pt[0],t.pt[1]);
            points.push_back(t);
        }
	//points.pop_back();
        input.close();
    }
	return points;   
}


void sort_points(TPointVector& pts, int lb, int ub, int depth) {
	int size = ub - lb + 1;
	if (size <= 4) {
		return;
	}

	int split = depth % DIMENSION;
	std::sort(pts.begin() + lb, pts.begin() + ub + 1, ComparePoints(split));
	int mid = (ub + lb) / 2;

	sort_points(pts, lb, mid, depth + 1);
	sort_points(pts, mid+1, ub, depth + 1);
}


int main ( int argc, char **argv ) {
#ifdef PAPI
	int retval, EventSet=PAPI_NULL;
	long long values[4] = {(long long)0, (long long)0, (long long)0, (long long)0};

	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if(retval != PAPI_VER_CURRENT)
		handle_error(retval);
	
	retval = PAPI_multiplex_init();
	if (retval != PAPI_OK) 
		handle_error(retval);
	retval = PAPI_create_eventset(&EventSet);
	if (retval != PAPI_OK) 
		handle_error(retval);
 
	// Add Total L3Cache Misses 
	retval = PAPI_add_event(EventSet, PAPI_L3_TCM);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// Total L3 cache accesses = total memory accesses. Needed for computing L3 miss rate. On Wabash, there are 3 layers of cache. 
	retval = PAPI_add_event(EventSet, PAPI_L3_TCA);
	if (retval != PAPI_OK) 
		handle_error(retval);

	retval = PAPI_set_multiplex(EventSet);
	if (retval != PAPI_OK) 
		handle_error(retval);

	 //TOTAL cycles 
	retval = PAPI_add_event(EventSet, PAPI_TOT_CYC);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// TOTAL instructions 
	retval = PAPI_add_event(EventSet, PAPI_TOT_INS);
	if (retval != PAPI_OK) 
		handle_error(retval);

#endif
	srand(time(NULL));
	boost::mpi::environment env(argc, argv);
	mpi::communicator world;
	std::string msg;
	mpi_process_group pg;
	int blkSize = BLOCK_SIZE;
	int numBuffers = NUM_PIPELINE_BUFFERS;
	int subtreeHeight = SUBTREE_HEIGHT;
	int blkIntakeLimit = BLOCK_INTAKE_LIMIT;
	std::vector<int> paramValues;

	int pid = process_id(pg);
	int numProcs = num_processes(pg);
	int replicationFactor = numProcs;
	GAL* gal = GAL::GAL_GetInstance(pg);
	
	CommLineOpts commLineOpts(argc, argv);
	
	TPointVector& points = ReadPoints(argv[1]);
	long int numPoints = points.size();

	TPointVector points2 = points;

	bool retstatus = commLineOpts.Get("BLOCK_SIZE", paramValues);
	if(retstatus)
		blkSize = paramValues[0];
	paramValues.clear();
	retstatus = commLineOpts.Get("SUBTREE_HEIGHT", paramValues);
	if(retstatus)
	subtreeHeight = paramValues[0];
	paramValues.clear();
	retstatus = commLineOpts.Get("BLOCK_INTAKE_LIMIT", paramValues);
	if(retstatus)
	blkIntakeLimit = paramValues[0];
	paramValues.clear();
	retstatus = commLineOpts.Get("REPLICATION", paramValues);
	if(retstatus)
		replicationFactor = paramValues[0];
	paramValues.clear();
	assert((replicationFactor !=0) && (replicationFactor <= numProcs));
	retstatus = commLineOpts.Get("NUM_PIPELINE_BUFFERS", paramValues);
	if(retstatus)
	numBuffers = paramValues[0];
	paramValues.clear();
	retstatus = commLineOpts.Get("PIPELINE_BUFFER_SIZES", paramValues);
	if(pid==0)
	{
#if !defined(MESSAGE_AGGREGATION)
		if(retstatus == true)
			printf("Warning: MESSAGE_AGGREGATION turned off. NUM_PIPELINE_BUFFERS option ignored. \n");
#endif
		printf("blkSize = %d subtreeHeight=%d blkIntakeLimit:%d numPipelineBuffers=%d\n",blkSize, subtreeHeight,blkIntakeLimit,numBuffers);
	}
#if 1

	//every process gets [startIndex, endIndex) chunk of rayVector.
	//Block distribution of points among processes
	long int startIndex = (pid) * (numPoints/replicationFactor);
	long int endIndex = (pid == (replicationFactor-1))?(numPoints):((pid+1) * (numPoints/replicationFactor));
	if(numPoints < numProcs)
	{
		if(pid >= numPoints)
		{
			startIndex = 0;
			endIndex = 0;
		}
		else
		{
			startIndex = pid;
			endIndex = pid+1;
		}
	}
	if(pid >= replicationFactor)
	{
		startIndex = 0;
		endIndex = 0;
	}

	(gal)->GAL_ConstructVPTree(points, subtreeHeight);
	//sort_points(points, 0, points.size()-1,0); this remains commented
	//std::sort(points2.begin(), points2.end(),ComparePoints(0));

	int randPtIndx = rand()%(endIndex-startIndex);
	/*std::cout<<"Point 191512 = ("<<points2[191512].pt[0]<<", "<<points2[191512].pt[1]<<", "<<points2[191512].pt[2]<<"): "<<std::endl;
	startIndex = randPtIndx;
	endIndex= startIndex+1;*/

	CorrelationVisitor cs(pid,points2, blkSize, blkIntakeLimit);
	cs.AssignWorkItemBounds(startIndex,endIndex, replicationFactor);
	cs.SetRadius(0.03);
	cs.ComputeNumberOfWorkBlocks();
	
	long int sum = 0, totalSum=0;
	long int sum_nodes_traversed=0, totalNodesTraversed=0;
	int status = STATUS_FAILURE;
	double startTime, endTime;
	if(pid == 0)
		printf("Traversal Started\n");
	
	synchronize(pg);
	startTime = clock();
#ifdef PAPI
	retval = PAPI_start(EventSet);
	if (retval != PAPI_OK) handle_error(retval);
#endif
	cs.VisitDFS(gal, pid, paramValues, numBuffers);

#ifdef PAPI
	/* Stop the counters */
	retval = PAPI_stop(EventSet, values);
	if (retval != PAPI_OK) 
		handle_error(retval);
#endif

	/*if(pid == 0)
	{
		for (int i=startIndex; i<endIndex; i++) 
		{
			std::cout<<i<<": ("<<points2[i].pt[0]<<", "<<points2[i].pt[1]<<"): "<<points2[i].nodesTraversed<<std::endl;
		}
	}*/
	
	for (int i=startIndex; i<endIndex; i++) 
	{
		sum+=points2[i].corr;
		sum_nodes_traversed +=points2[i].nodesTraversed;
	}
	endTime=clock();
	double maxConsumedTime, consumedTime = endTime - startTime;
	/*//if(pid == 0)
	{
	std::cout<<process_id(pg)<< ": Sum Nodes traversed :"<<sum_nodes_traversed<<std::endl;
	//std::cout<<"time consumed: "<<consumedTime/CLOCKS_PER_SEC<<" seconds"<<std::endl;
	}*/
	
	reduce(world,sum, totalSum, std::plus<long int>(),0);
	reduce(world,sum_nodes_traversed, totalNodesTraversed, std::plus<long int>(),0);
	reduce(world,consumedTime, maxConsumedTime, boost::parallel::maximum<double>(),0);
	if(pid == 0)
	{
		printf("%d: KdTree correlation: %f\n",pid,(((float)totalSum)/numPoints));
		std::cout<<": Sum Nodes traversed :"<<totalNodesTraversed<<std::endl;
		std::cout<<"time consumed: "<<maxConsumedTime/CLOCKS_PER_SEC<<" seconds"<<std::endl;
	}
	gal->GAL_PrintGraph();
#ifdef TRAVERSAL_PROFILE
	/*std::ofstream vertDistStream;
	char vertDistLog[32];
	sprintf(vertDistLog,"vertDist.txt",pid);
	vertDistStream.open(vertDistLog, std::fstream::out);
	std::vector<int>::iterator tmpVecIter;
	std::map<int,std::vector<int> > vertDistMap;
	int maxTraversalLen=0;
	for (int i=startIndex; i<endIndex; i++) 
	{
		std::sort((points2)[i].visitedNodes.begin(),(points2)[i].visitedNodes.end());
		if((points2)[i].visitedNodes.size() > maxTraversalLen)
			maxTraversalLen = (points2)[i].visitedNodes.size();
		tmpVecIter =(points2)[i].visitedNodes.begin();
		for(;tmpVecIter!=(points2)[i].visitedNodes.end();tmpVecIter++)
		{
			vertDistStream<<*tmpVecIter<<" ";
		}
		int eofMarker=-1;
		vertDistStream<<eofMarker<<std::endl;
	}
	printf("Traversal similarity distribution and truncation pattern log stored in vertDist.txt\n");
	vertDistStream.close();*/
	/*Point* randPt = &(points2[randPtIndx]);
	std::vector<int>::iterator depthIter = randPt->nodeDepthAtTruncation.begin();
	for(;depthIter!=randPt->nodeDepthAtTruncation.end();depthIter++)
	{
		printf("%d\n",*depthIter);
	}
	printf("\nCollected truncation depth of point %d . Total # truncations %d\n",randPtIndx, randPt->nodeDepthAtTruncation.size());*/

#endif

#ifdef PAPI
	float avgMissRate, missRate = values[0]/(double)(values[1]);
	float avgCPI, CPI = values[2]/(double)(values[3]);
	reduce(world,CPI, avgCPI, std::plus<float>(),0);
	reduce(world,missRate, avgMissRate, std::plus<float>(),0);
	if(pid==0)
		printf("Average L3 Miss Rate:%f Average CPI (Total Cycles/ Total Instns): (%ld/%ld) = %f\n",avgMissRate/numProcs,values[2],values[3],avgCPI/numProcs);
		//printf("Total LD/ST Instns: %ld\n",values[0]);
	//printf("%d: L2 Cache Miss Rate:%f CPI:%f\n",pid, values[0]/(double)(values[1]),values[2]/(double)(values[3]));

#endif
    	//==========================================================================
  return 0;
#endif
}

void CommLineOpts::ParseCommLineOpts(int argc, char **argv, char c)
{
	int i=2;
	//std::cout<<"Command Line Options"<<std::endl;
	while(argc > 2)
	{
		char* str = argv[i];
		char* tmpStr = str;
		int substrLen=0;

		while(*str && *str != c)
		{
			substrLen++;
			str++;
		}
		std::string paramName(tmpStr,substrLen);
		std::string paramVal(++str);
		//int paramVal=atoi(++str);
		//std::cout<<paramName<<" = "<<paramVal<<std::endl;
		//paramList.push_back(std::make_pair(paramName, paramVal));
		paramList.push_back(std::make_pair(paramName, paramVal));
		i++;
		argc--;
	} 
}

//bool CommLineOpts::Get(const std::string& paramName, int & val)
bool CommLineOpts::Get(const std::string& paramName, std::vector<int> & val)
	{
		bool flag = false;
		clopts::iterator iter=paramList.begin();
		while(iter != paramList.end())
		{
			if(!((iter->first).compare(paramName)))
			{
				//val = iter->second;
				if(!(paramName.compare("PIPELINE_BUFFER_SIZES")))
				{
					ParseCommaSeperatedValues((iter->second).c_str(),val);
				}
				else
				{
					val.push_back(atoi((iter->second).c_str()));	
				}
				flag = true;
				break;
			}
			iter++;
		}	
		return flag;
	}

using namespace std;

void CommLineOpts::ParseCommaSeperatedValues(const char *str, std::vector<int>& retVal)
{
    do
    {
        const char *startStr = str;

        while(*str != ',' && *str)
            str++;
	std::string tmpStr(string(startStr, str));
        retVal.push_back(atoi(tmpStr.c_str()));
    } while (0 != *str++);
}

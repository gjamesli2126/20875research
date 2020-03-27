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
#include "AuxFn.h"
#include <boost/mpi.hpp>

#define BATCH_SIZE 1000000
int batchSize=BATCH_SIZE;

#ifdef DYNAMIC_LB
int DYNAMIC_LB_TRIGGER=512;
int STATIC_LB_TRIGGER=4;
#endif
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

	bool Get(const std::string& paramName, std::vector<int>& val);
	void ParseCommaSeperatedValues(const char *str, std::vector<int>& retVal);
};


int uniqId;

uint64_t nbodies=0;
uint64_t ntimesteps=0;
float dtime;
float eps;
float tol;


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
	
	// Total L1 cache accesses = total memory accesses. Needed for computing L3 miss rate. On Qstruct, there are 2 layers of cache. 
	retval = PAPI_add_event(EventSet, PAPI_L3_TCA);
	if (retval != PAPI_OK) 
		handle_error(retval);

	retval = PAPI_set_multiplex(EventSet);
	if (retval != PAPI_OK) 
		handle_error(retval);

	// TOTAL cycles 
	retval = PAPI_add_event(EventSet, PAPI_TOT_CYC);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// TOTAL instructions 
	retval = PAPI_add_event(EventSet, PAPI_TOT_INS);
	if (retval != PAPI_OK) 
		handle_error(retval);

#endif

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
	retstatus = commLineOpts.Get("BATCH_SIZE", paramValues);
	if(retstatus)
		batchSize = paramValues[0];
	paramValues.clear();
#ifdef DYNAMIC_LB
	retstatus = commLineOpts.Get("DYNAMIC_LB_TRIGGER", paramValues);
	if(retstatus)
		DYNAMIC_LB_TRIGGER = paramValues[0];
	paramValues.clear();
	retstatus = commLineOpts.Get("STATIC_LB_TRIGGER", paramValues);
	if(retstatus)
		STATIC_LB_TRIGGER = paramValues[0];
	paramValues.clear();
#endif

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

	int rdPtrIndx=0;
	Point* pointArr; 
	pointArr=NULL;
	std::ifstream input(argv[1], std::fstream::in);
	if(input.fail())
	{
		std::cout<<"File does not exist. exiting"<<std::endl;
		MPI_Finalize();
		exit(0);
	}
	input >> nbodies;
	input >> ntimesteps;
	input >> dtime;
	input >> eps;
	input >> tol;


	long int numPoints = atol(argv[2]);
#ifdef HYBRID_BUILD	
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
#else
	long int startIndex = 0;
	long int endIndex = numPoints;
#endif


	//printf("%d startIndex:%ld endIndex:%ld\n",pid,startIndex, endIndex);

	pointArr = new Point[endIndex-startIndex];
	int numPointsPerProcess=endIndex-startIndex;
	ReadPoints(input, startIndex, endIndex, pointArr);
	double startTime, endTime;
	startTime=clock();
	gal->GAL_ConstructKDTree(pointArr, numPointsPerProcess, false, subtreeHeight);
	endTime=clock();
	if(pid == 0)
		printf("Tree construction time:%f secs\n",(endTime-startTime)/CLOCKS_PER_SEC);
	delete [] pointArr;
        input.close();
	
#if 1
	input.open(argv[1],std::fstream::in);
	if(input.fail())
	{
		std::cout<<"Error opening file. exiting"<<std::endl;
		MPI_Finalize();
		exit(0);
	}
	input >> nbodies;
	input >> ntimesteps;
	input >> dtime;
	input >> eps;
	input >> tol;

	if(numPoints < batchSize)
		batchSize = numPoints;
	pointArr = new Point[batchSize];
	int batchNumber=0;
	long int sum_nodes_traversed=0, totalNodesTraversed=0;
	bool hasMoreData = true;
	srand(time(NULL));
	double totalTimeConsumed = 0;
	do
	{
		batchNumber++;
		memset(pointArr,0,sizeof(Point)*batchSize);
		hasMoreData = ReadPointsInBatch(input, pointArr, numPoints, batchSize);
		startIndex = (pid) * (batchSize/replicationFactor);
		endIndex = (pid == (replicationFactor-1))?(batchSize):((pid+1) * (batchSize/replicationFactor));
		if(batchSize < numProcs)
		{
			if(pid >= batchSize)
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
		/*int randPtIndx = 289383;//rand()%(endIndex-startIndex);
std::cout<<"Point = ("<<pointArr[randPtIndx].cofm.x<<", "<<pointArr[randPtIndx].cofm.y<<", "<<pointArr[randPtIndx].cofm.z<<"): "<<std::endl;

		startIndex = randPtIndx;
		endIndex= startIndex+1;*/

		CorrelationVisitor cs(pid, pointArr, blkSize, blkIntakeLimit);
		cs.AssignWorkItemBounds(startIndex,endIndex, replicationFactor);
		cs.ComputeNumberOfWorkBlocks();
		cs.SetSimulationParams(dtime,eps,tol,ntimesteps);

		int status = STATUS_FAILURE;
		synchronize(pg);
		if(pid == 0)
			printf("Traversal Started. Batch %d.\n",batchNumber);
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
		for (int i=startIndex; i<endIndex; i++) 
		{
			sum_nodes_traversed +=(pointArr)[i].nodesTraversed;
				//std::cout<<i<<": ("<<pointArr[i].pt[0]<<", "<<pointArr[i].pt[1]<<"): "<<pointArr[i].nodesTraversed<<std::endl;
		}
		endTime=clock();
		double maxConsumedTime, consumedTime = endTime - startTime;
		reduce(world,consumedTime, maxConsumedTime, boost::parallel::maximum<double>(),0);
		totalTimeConsumed += maxConsumedTime/CLOCKS_PER_SEC;
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
		std::sort((pointArr)[i].visitedNodes.begin(),(pointArr)[i].visitedNodes.end());
		if((pointArr)[i].visitedNodes.size() > maxTraversalLen)
			maxTraversalLen = (pointArr)[i].visitedNodes.size();
		tmpVecIter =(pointArr)[i].visitedNodes.begin();
		for(;tmpVecIter!=(pointArr)[i].visitedNodes.end();tmpVecIter++)
		{
			vertDistStream<<*tmpVecIter<<" ";
		}
		int eofMarker=-1;
		vertDistStream<<eofMarker<<std::endl;
	}
	printf("Traversal similarity distribution and truncation pattern log stored in vertDist.txt\n");
	vertDistStream.close();*/
	/*Point* randPt = &(pointArr[randPtIndx]);
	std::vector<int>::iterator depthIter = randPt->nodeDepthAtTruncation.begin();
	for(;depthIter!=randPt->nodeDepthAtTruncation.end();depthIter++)
	{
		printf("%d\n",*depthIter);
	}
	printf("\nCollected truncation depth of point %d . Total # truncations %d\n",randPtIndx, randPt->nodeDepthAtTruncation.size());*/
#endif

	}while(hasMoreData);

	delete [] pointArr;
        input.close();
	reduce(world,sum_nodes_traversed, totalNodesTraversed, std::plus<long int>(),0);
	if(pid==0)
	{
		printf("time consumed: %f seconds\n",totalTimeConsumed);
		printf(": Sum Nodes traversed: %ld\n",totalNodesTraversed);
	}
#endif
	gal->GAL_PrintGraph();

#ifdef PAPI
	float avgMissRate, missRate = values[0]/(double)(values[1]);
	float avgCPI, CPI = values[2]/(double)(values[3]);
	reduce(world,missRate, avgMissRate, std::plus<float>(),0);
	reduce(world,CPI, avgCPI, std::plus<float>(),0);
	if(pid==0)
		printf("Average L3 Miss Rate:%f Average CPI:%f\n",avgMissRate/numProcs,avgCPI/numProcs);
	//printf("%d:L3 Cache Miss Rate:%f CPI:%f\n",pid, missRate,CPI);

#endif

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
		//std::cout<<paramName<<" = "<<paramVal<<std::endl;
		paramList.push_back(std::make_pair(paramName, paramVal));
		i++;
		argc--;
	} 
}

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

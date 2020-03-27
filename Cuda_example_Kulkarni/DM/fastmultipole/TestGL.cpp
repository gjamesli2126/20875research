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

#define BATCH_SIZE 131072
int batchSize=BATCH_SIZE;

#ifdef SPAD_2
int numReplicatedSubtrees=0;
#endif
#ifdef PAPI1
#include"papi.h"
void handle_error(int retval)
{
	printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	exit(1);
}
#endif

Box random_particles(Point* particles, int m);
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


int main ( int argc, char **argv ) {

#ifdef PAPI1
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
#ifdef SPAD_2
	retstatus = commLineOpts.Get("SPAD2", paramValues);
	if(retstatus)
		numReplicatedSubtrees = paramValues[0];
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
	srand(0);
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
        input.close();

	if(argc < 3)
	{
		printf("usage: ./FMM <inputfile(currently ignored)> numpoints BATCH_SIZE=<optional> SUBTREE_HEIGHT=64\n");
		return 0;
	}
	long int numPoints = atol(argv[2]);
	Box box;
	box = random_particles(pointArr,numPoints);
	int boxHeight = box.endY - box.startY;
	int boxWidth = box.endX - box.startX;
	if(boxWidth > boxHeight)
		box.endY = box.startY + boxWidth;
	else
		box.endX = box.startX + boxHeight;

	srand(0);
	double startTime, endTime;
	startTime = clock();
	if(numPoints >= batchSize)
	{
		long int tmpPoints = 0;
		while(tmpPoints < numPoints)
		{
			int curBatchSize = ((tmpPoints+batchSize)<numPoints)?batchSize:(numPoints - tmpPoints);
			pointArr = new Point[curBatchSize];
			random_particles(pointArr,curBatchSize);
			tmpPoints += curBatchSize;		
			bool hasMoreData = (tmpPoints < numPoints)?true:false;
			gal->GAL_ConstructQuadTree(pointArr, box, curBatchSize, hasMoreData, subtreeHeight);
			delete [] pointArr;
		}
	}
	endTime = clock();
	double consumedTime = endTime - startTime;
	if(pid == 0)
	{
		printf("time consumed: %f seconds\n",consumedTime/CLOCKS_PER_SEC);
	}
	
#if 0
	input.open(argv[1],std::fstream::in);
	if(input.fail())
	{
		std::cout<<"Error opening file. exiting"<<std::endl;
		MPI_Finalize();
		exit(0);
	}
	
	if(numPoints < batchSize)
		batchSize = numPoints;
	pointArr = new Point[batchSize];
	int batchNumber=0;
	long int sum = 0, totalSum=0;
	long int sum_nodes_traversed=0, totalNodesTraversed=0;
	bool hasMoreData = true;
	srand(time(NULL));

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
		//qsort(pointArr, numPoints, sizeof(Point), compare_point);
		CorrelationVisitor cs(pid, pointArr, blkSize, blkIntakeLimit);
		cs.AssignWorkItemBounds(startIndex,endIndex, replicationFactor);
		cs.ComputeNumberOfWorkBlocks();

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
		endTime=clock();
		double maxConsumedTime, consumedTime = endTime - startTime;
		reduce(world,consumedTime, maxConsumedTime, boost::parallel::maximum<double>(),0);
		if(pid == 0)
		{
			printf("time consumed: %f seconds\n",maxConsumedTime/CLOCKS_PER_SEC);
		}
	}while(hasMoreData);

	delete [] pointArr;
        input.close();
	reduce(world,sum, totalSum, std::plus<long int>(),0);
	reduce(world,sum_nodes_traversed, totalNodesTraversed, std::plus<long int>(),0);
	if(pid==0)
	{
		printf("%d: KdTree correlation: %f\n",pid,(((float)totalSum)/numPoints));
		printf(": Sum Nodes traversed: %ld\n",totalNodesTraversed);
	}
	gal->GAL_PrintGraph();
#endif
#ifdef PAPI1
	float avgMissRate, missRate = values[0]/(double)(values[1]);
	float avgCPI, CPI = values[2]/(double)(values[3]);
	reduce(world,missRate, avgMissRate, std::plus<float>(),0);
	reduce(world,CPI, avgCPI, std::plus<float>(),0);
	if(pid==0)
		printf("Average L3 Miss Rate:%f Average CPI (Total Cycles/ Total Instns): (%ld/%ld) = %f\n",avgMissRate/numProcs,values[2],values[3],avgCPI/numProcs);
		//printf("Total LD/ST Instns: %ld \n",values[0]);

	//printf("%d:L3 Cache Miss Rate:%f CPI:%f\n",pid, missRate,CPI);

#endif

  return 0;
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

Box random_particles(Point* particles, int m)
{
	Box ret;
	if(particles)
     	{
		int labelCount=0;
	     	for (int i =0 ; i < m ; i ++ )
       		{ 
	       		particles[i].coordX = (rand() % 128 ) ; 
			particles[i].coordY = (rand() % 128 ) ; 

			if(particles[i].coordX < ret.startX)
				ret.startX = particles[i].coordX;
			if(particles[i].coordX > ret.endX)
				ret.endX = particles[i].coordX;
			if(particles[i].coordY < ret.startY)
				ret.startY = particles[i].coordY;
			if(particles[i].coordY > ret.endY)
				ret.endY = particles[i].coordY;

			particles[i].mass = 2;
			particles[i].potential =0.;
			particles[i].id = labelCount++;
	       }
	}
	else
	{
	     	for (int i =0 ; i < m ; i ++ )
		{
			int coordX = (rand() % 128 ) ; 
			int coordY = (rand() % 128 ) ; 

			if(coordX < ret.startX)
				ret.startX = coordX;
			if(coordX > ret.endX)
				ret.endX = coordX;
			if(coordY < ret.startY)
				ret.startY = coordY;
			if(coordY > ret.endY)
				ret.endY = coordY;
		}

	}

	return ret;
	       
}



/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
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
#include "SPIRIT.hpp"
#include "PCKDTreeVisitor.hpp"
#include "PCKdtreeTypes.hpp"
#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<boost/graph/distributed/adjacency_list.hpp>

int main ( int argc, char **argv ) {
	if(argc <4)
	{
		printf("usage:./KD <inputfile> <numpoints> <radius> <SPIRIT opts (optional)>...\n");
		return 0;
	}
	boost::mpi::environment env(argc, argv);
	mpi::communicator world;
	mpi_process_group pg;
	int pid = process_id(pg);
	int numProcs = num_processes(pg);
	double startTime, endTime, totalTime=0.;
	std::vector<Point*> pointArr;
	if(pid == 0)
		printf("usage:./KD <inputfile> <numpoints> <radius> <SPIRIT opts (optional)>...\n");

	long int numPoints = atol(argv[2]);
	InputFileParser* parser = new PCKDTreeInputFileParser(PCKDTREE);
	Optimizations opts(argc, argv);
	SPIRIT* spirit = SPIRIT::GetInstance(opts,pg);
	spirit->ReadTreeData(argv[1],numPoints, pointArr, parser);
	long int numPointsRead = pointArr.size();
	if(numPointsRead < 0)
	{
		MPI_Finalize();
		printf("Error reading data\n");
		exit(0);
	}
	spirit->ConstructBinaryTree(pointArr, numPoints, parser);
	pointArr.clear();
	double rad = atof(argv[3]);
	if(pid==0)
		printf("radius %lf\n",rad);
	PCKDTreeVisitor* vis = new PCKDTreeVisitor(rad);
	spirit->Traverse(vis); 	
	spirit->PrintResults();
	spirit->PrintGraph();
	spirit->ResetInstance();

	long int totalSum;
	long int corr=vis->GetCorr();
	reduce(world,corr, totalSum, std::plus<long int>(),0);
	if(pid==0)
	{
		printf("KdTree correlation: %f\n",(((float)totalSum)/numPoints));
	}
    	//==========================================================================
  return 0;
}



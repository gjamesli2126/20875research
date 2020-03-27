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
#include "KDTreeVisitor.hpp"
#include "KdtreeTypes.hpp"
#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<boost/graph/distributed/adjacency_list.hpp>

int main ( int argc, char **argv ) {
	boost::mpi::environment env(argc, argv);
	mpi::communicator world;
	mpi_process_group pg;
	int pid = process_id(pg);
	int numProcs = num_processes(pg);
	double startTime, endTime, totalTime=0.;
	std::vector<Point*> pointArr;
	long int numPoints = atol(argv[2]);
	InputFileParser* parser = new KDTreeInputFileParser(KDTREE);
	Optimizations opts(argc, argv);
	SPIRIT* spirit = SPIRIT::GetInstance(opts,pg);
	spirit->ReadTreeData(argv[1],numPoints/2, pointArr, parser);
	long int numPointsRead = pointArr.size();
	if(numPointsRead < 0)
	{
		MPI_Finalize();
		printf("Error reading data\n");
		exit(0);
	}
	spirit->ConstructBinaryTree(pointArr, numPoints, parser);
	pointArr.clear();
	KDTreeVisitor* vis = new KDTreeVisitor();
	spirit->Traverse(vis); 	
	spirit->PrintResults();
	spirit->PrintGraph();
	spirit->ResetInstance();
    	//==========================================================================
  return 0;
}



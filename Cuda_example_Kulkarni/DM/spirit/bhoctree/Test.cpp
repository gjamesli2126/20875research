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
#include "OctreeVisitor.hpp"
#include "OctreeTypes.hpp"
#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<boost/graph/distributed/adjacency_list.hpp>
#ifdef PAPI
#include"papi.h"
void handle_error(int retval)
{
	printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	exit(1);
}
#endif

//long int pointsVisited;

int main ( int argc, char **argv ) {
	boost::mpi::environment env(argc, argv);
	mpi::communicator world;
	mpi_process_group pg;
	int pid = process_id(pg);
	int numProcs = num_processes(pg);
	double startTime, endTime, totalTime=0.;
	long int numPoints;
	int nTimeSteps;
	float dTime, eps, tol;
	std::ifstream input(argv[1], std::fstream::in);
	if(input.fail())
	{
		std::cout<<"File does not exist. exiting"<<std::endl;
		MPI_Finalize();
		exit(0);
	}
	input >> numPoints;
	input >> nTimeSteps;
	input >> dTime;
	input >> eps;
	input >> tol;
	input.close();

	InputFileParser* parser = new OctreeInputFileParser(OCTREE);
	Optimizations opts(argc, argv);
	SPIRIT* spirit = SPIRIT::GetInstance(opts,pg);
	float dia = spirit->ConstructOctree(argv[1],numPoints,parser);
	OctreeVisitor* vis = new OctreeVisitor(dTime, eps, tol, dia);
	spirit->Traverse(vis); 	
	delete parser;
	spirit->PrintResults();
	spirit->PrintGraph();
	spirit->ResetInstance();
  	return 0;
}



/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef OPTIMIZATIONS
#define OPTIMIZATIONS
#include<string>
#include<vector>
/*Default values of parameters. They can also be defined at runtime.
 *====================================================*/
/* Message aggregation is a feature that buffers the blocks at pseudo leaves and sends them to the children of those pseudo roots only when the buffer fills up or when 
 * there are no more input blocks left to traverse. */
//#define MESSAGE_AGGREGATION //uncomment these only if the 'make pipelined' option is not defined in makefile.
//#define SPAD_2
#define BLOCK_SIZE 1024 
#define BLOCK_INTAKE_LIMIT 1024
#define SUBTREE_HEIGHT 10
#define BATCH_SIZE 1000000 //max size of shared memory
#define DEFAULT_REP_PERCENTAGE 0
#define MAX_POINTS_PER_CELL 16

/*====================================================*/
#ifdef MESSAGE_AGGREGATION
#define DEFAULT_BUFFER_SIZE 256
#define PIPELINE_BUFFER_SIZE_LEVEL_1 4096
#endif

typedef enum WorkDist{BLOCK_DIST, BLOCK_CYCLIC}WorkDist;

using namespace std;
class Optimizations
{
public:
	int blockSize;
	int subtreeHeight; 
	int blkIntakeLimit; 
	int batchSize;
	char replication;
	short int maxPointsPerCell;
	WorkDist distribution;
	bool pipelined;
#ifdef MESSAGE_AGGREGATION
	int pipelineBufferSize;
#endif
	typedef std::vector<std::pair<std::string, std::string> > clopts;
	clopts paramList;
	void ParseOptimizations(int argc, char **argv, char c='=');
	Optimizations(int argc, char **argv)
	{
		pipelined=false;
		distribution = BLOCK_DIST;
		blockSize = BLOCK_SIZE;
		batchSize = BATCH_SIZE;
		blkIntakeLimit = BLOCK_INTAKE_LIMIT;
		subtreeHeight=SUBTREE_HEIGHT;
		replication = DEFAULT_REP_PERCENTAGE;
		maxPointsPerCell = MAX_POINTS_PER_CELL;
#ifdef MESSAGE_AGGREGATION
		pipelineBufferSize=PIPELINE_BUFFER_SIZE_LEVEL_1;
#endif
		ParseOptimizations(argc,argv,'=');
	}

	bool Get(const std::string& paramName, std::vector<int>& val);
	//void ParseCommaSeperatedValues(const char *str, std::vector<int>& retVal);
};


#endif

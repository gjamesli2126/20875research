#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<iostream>
#include<vector>
#include "clops.h"

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

	std::vector<int> paramValues;
	bool retstatus = Get("BLOCK_SIZE", paramValues);
	if(retstatus)
		blkSize = paramValues[0];
	else
		blkSize = BLOCK_SIZE;
	paramValues.clear();

	retstatus = Get("SUBTREE_HEIGHT", paramValues);
	if(retstatus)
		subtreeHeight = paramValues[0];
	else
		subtreeHeight = SUBTREE_HEIGHT;
	paramValues.clear();

	retstatus = Get("BLOCK_INTAKE_LIMIT", paramValues);
	if(retstatus)
		blkIntakeLimit = paramValues[0];
	else
		blkIntakeLimit = BLOCK_INTAKE_LIMIT;
	paramValues.clear();
	
	retstatus = Get("DGENTREE_VCOUNT", paramValues);
	if(retstatus)
		dgentree_vcount = paramValues[0];
	else
		dgentree_vcount = DGENTREE_VCOUNT;
	paramValues.clear();

#ifdef SPAD_2
	retstatus = Get("SPAD2", paramValues);
	if(retstatus)
		numReplicatedSubtrees = paramValues[0];
	else
		numReplicatedSubtrees = 0;
	paramValues.clear();
#endif
	retstatus = Get("NUM_PIPELINE_BUFFERS", paramValues);
#if !defined(MESSAGE_AGGREGATION)
	if(retstatus == true)
			printf("Warning: MESSAGE_AGGREGATION turned off. NUM_PIPELINE_BUFFERS option ignored. \n");
#endif

#ifdef MESSAGE_AGGREGATION
	if(retstatus)
		numBuffers = paramValues[0];
	else
		numBuffers = NUM_PIPELINE_BUFFERS;
	paramValues.clear();
	for(int i=0;i<3;i++)
	{
		if(i == 0)
			pipelineBufferSizes.push_back(PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,1));
		else if (i == 1)
			pipelineBufferSizes.push_back(PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,2));
		else if (i == 2)
			pipelineBufferSizes.push_back(PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,3));
	}
#endif

	retstatus = Get("PIPELINE_BUFFER_SIZES", paramValues);
#if !defined(MESSAGE_AGGREGATION)
	/*if(retstatus == true)
			printf("Warning: MESSAGE_AGGREGATION turned off. PIPELINE_BUFFER_SIZES option ignored. \n");*/
#endif
#ifdef MESSAGE_AGGREGATION
	if(retstatus)
	{
		for(int i=0;i<paramValues.size();i++)
		{
			if(i < 3)
				pipelineBufferSizes[i] = paramValues[i];
		}
	}
#endif
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

void CommLineOpts::PrintOpts()
{
#ifdef SPAD_2
	printf("spad2: %d BlockSize:%d SubtreeHeight:%d BlockIntakeLimit:%d\n",numReplicatedSubtrees,blkSize,subtreeHeight,blkIntakeLimit);
#else
	printf("BlockSize:%d SubtreeHeight:%d BlockIntakeLimit:%d\n",blkSize,subtreeHeight,blkIntakeLimit);
#endif
}

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

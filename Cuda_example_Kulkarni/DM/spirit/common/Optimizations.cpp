/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include"Optimizations.hpp"
#include<stdlib.h>
void Optimizations::ParseOptimizations(int argc, char **argv, char c)
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
		blockSize = paramValues[0];
	paramValues.clear();
	retstatus = Get("SUBTREE_HEIGHT", paramValues);
	if(retstatus)
	subtreeHeight = paramValues[0];
	paramValues.clear();
	retstatus = Get("BLOCK_INTAKE_LIMIT", paramValues);
	if(retstatus)
	blkIntakeLimit = paramValues[0];
	paramValues.clear();
	retstatus = Get("BATCH_SIZE", paramValues);
	if(retstatus)
		batchSize = paramValues[0];
	paramValues.clear();
	retstatus = Get("DIST", paramValues);
	if(retstatus)
		distribution = static_cast<WorkDist>(paramValues[0]);
	paramValues.clear();
	retstatus = Get("REPLICATION", paramValues);
	if(retstatus)
	replication = paramValues[0];
	paramValues.clear();
	retstatus = Get("MAX_POINTS_PER_CELL", paramValues);
	if(retstatus)
	maxPointsPerCell = paramValues[0];
	paramValues.clear();
	int pipe=0;
	retstatus = Get("PIPELINED", paramValues);
	if(retstatus)
	{
		pipe = paramValues[0];
		pipelined=(pipe==1)?true:false;
	}
	paramValues.clear();


#ifdef MESSAGE_AGGREGATION
	retstatus = Get("PIPELINE_BUFFER_SIZES", paramValues);
	if(retstatus)
		pipelineBufferSize = paramValues[0];
	paramList.clear();
#endif
}

//bool Optimizations::Get(const std::string& paramName, int & val)
bool Optimizations::Get(const std::string& paramName, std::vector<int> & val)
	{
		bool flag = false;
		clopts::iterator iter=paramList.begin();
		while(iter != paramList.end())
		{
			if(!((iter->first).compare(paramName)))
			{
				val.push_back(atoi((iter->second).c_str()));	
				flag = true;
				break;
			}
			iter++;
		}	
		return flag;
	}

using namespace std;

/*void Optimizations::ParseCommaSeperatedValues(const char *str, std::vector<int>& retVal)
{
    int i=0;
    do
    {
        const char *startStr = str;

        while(*str != ',' && *str)
            str++;
	std::string tmpStr(string(startStr, str));
        retVal.push_back(atoi(tmpStr.c_str()));
    } while (0 != *str++);
}*/

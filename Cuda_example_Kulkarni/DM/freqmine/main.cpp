/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <cassert>
#include <pthread.h>
#include<cstdlib>
#include <iostream>
#include <set>
#include <vector>
#include<fstream>
#include<sstream>
#include<string.h>
#include<sys/time.h>
#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<boost/graph/parallel/algorithm.hpp>

using namespace boost;
using boost::graph::distributed::mpi_process_group;
#include "fptree.hpp"
#ifdef PAPI
#include"papi.h"
void handle_error(int retval)
{
	printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	exit(1);
}
#endif

unsigned spliceDepth; 
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

	bool Get(const std::string& paramName, void* val);
	void ParseCommaSeperatedValues(const char *str, std::vector<int>& retVal);
};


void ReadTransactionDB(std::vector<Transaction>& transactions, const char* fileName, int procRank);
void ParallelFPTreeMining(void * inParam, int start, int end, void* outParam);

#ifdef DEBUG
void PrintTreeStats(FPNode* node, std::map<char, int>& nodesAtVertices);
#endif

int get_elapsed_usec(struct timeval& start_time, struct timeval& end_time) 
{
	int sec = end_time.tv_sec - start_time.tv_sec;
	int usec = end_time.tv_usec - start_time.tv_usec;
	return (sec * 1000000) + usec;
}

typedef void (*thread_function)(void* funcIn, int start, int end, void* funcOut);
struct targs {
	thread_function func;
	int start, end;
	void *funcIn, *funcOut;
};

static void *thread_entry(void * arg) 
{
	struct targs * t = (struct targs*)arg;
	t->func(t->funcIn, t->start, t->end, t->funcOut);
}

int *distribute_among(int initStart, int end, int numt) 
{
	int *ret = new int[numt * 2];
	int start = initStart;
	int i;
	int num;
	for (i = 0; i < numt; i++) {
		num = (end - start) / (numt - i);
		ret[2 * i] = start;
		ret[2 * i + 1] = start + num;
		start += num;
	}
	return ret;
}



int main(int argc, char **argv) {
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
 
	// Add Total L2Cache Misses 
	retval = PAPI_add_event(EventSet, PAPI_L3_TCM);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// Total L1 cache accesses = total memory accesses. Needed for computing L2 miss rate. On Qstruct, there are 2 layers of cache. 
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
	mpi_process_group pg;

	int procRank = process_id(pg);
	int numProcs = num_processes(pg);
	struct timeval start_time, end_time;
	std::vector<Transaction> transactions;
    	unsigned minimum_support_treshold;
	char transactionDBName[255];
	memset(transactionDBName,0,255);
	CommLineOpts commLineOpts(argc, argv);
	int numThreads;

	bool retstatus = commLineOpts.Get("SPLICE_DEPTH", &spliceDepth);
	if(!retstatus)
		spliceDepth = SPLICE_DEPTH;//default spice depth
	retstatus = commLineOpts.Get("MIN_SUPPORT", &minimum_support_treshold);
	if(!retstatus)
		minimum_support_treshold = SUPP_THRESHOLD;
	retstatus = commLineOpts.Get("NUM_THREADS", &numThreads);
	if(!retstatus)
		numThreads = NUM_THREADS;
	retstatus = commLineOpts.Get("TRANSACTION_DB", transactionDBName);
	if(!retstatus)
	{
		printf("No transaction database file argument specified! TRANSACTION_DB=<name>\n");
		return 0;
	}
	if(procRank == 0)
		printf("Parameters used: SPLICE_DEPTH = %d MIN_SUPPORT=%d TRANSACTION_DB=%s NUM_THREADS=%d\n",spliceDepth, minimum_support_treshold,transactionDBName,numThreads);

	//minimum_support_treshold = atoi(argv[1]);
	ReadTransactionDB(transactions, transactionDBName, procRank);
	gettimeofday(&start_time, NULL);
	FPTree* fptree=new FPTree(transactions, minimum_support_treshold);
	gettimeofday(&end_time, NULL);
	long int consumedTime = get_elapsed_usec(start_time, end_time);
	long int maxConsumedTime;
	reduce(world,consumedTime, maxConsumedTime, boost::parallel::maximum<long int>(),0);
	if(procRank == 0)
		printf("tree construction time: %f seconds\n",maxConsumedTime/(float)CLOCKS_PER_SEC);
#ifdef DEBUG
	std::map<char, int> nodesAtHeight;
	PrintTreeStats(fptree.root, nodesAtHeight);
	std::map<char,int>::iterator mIter=nodesAtHeight.begin();
	for(;mIter!=nodesAtHeight.end();mIter++)
		printf("VerticesAtHeight %d : %d\n",mIter->first,mIter->second);
#endif
	//std::cout<<"Constructing fptree done. Mining fptree..."<<std::endl;
#ifdef PAPI
			retval = PAPI_start(EventSet);
			if (retval != PAPI_OK) handle_error(retval);
#endif
	gettimeofday(&start_time, NULL);
	TreesPendingMining treesToBeMined;
	Item nullPrefix;
    	//std::set<Pattern> patterns = fptree_growth( fptree, 0, spliceDepth, nullPrefix, treesToBeMined );
    	std::set<Pattern> patterns, remainingPatterns;
    	fptree_growth( fptree, 0, spliceDepth, nullPrefix, treesToBeMined, patterns);
	delete fptree;
	//printf("mined patterns: %d, Pending trees to be mined:%d\n",patterns.size(), treesToBeMined.size());
	int noDuplicates=0;
	int numMinedPatterns=0;
	
	int numPoints = treesToBeMined.size();
	int startIndex = procRank * (numPoints/numProcs);
	int endIndex = (procRank == (numProcs - 1))?numPoints:((procRank+1) * (numPoints/numProcs));
	if(numPoints < numProcs)
	{
		if(procRank >= numPoints)
		{
			startIndex = 0;
			endIndex = 0;
		}
		else
		{
			startIndex = procRank;
			endIndex = procRank+1;
		}
	}

	//printf("%d startIndex %d endindex %d\n",procRank, startIndex, endIndex);
	ParallelFPTreeMining(&treesToBeMined, startIndex, endIndex, &remainingPatterns);
	gettimeofday(&end_time, NULL);
	numMinedPatterns = remainingPatterns.size();
	if(procRank == 0)
		numMinedPatterns += patterns.size();
		
	int totalMinedPatterns=0;
	all_reduce(world,numMinedPatterns, totalMinedPatterns, std::plus<int>());
	//printf("Total number of mined patterns:%d\n",patterns.size());
	if(procRank == 0)
		printf("%d Total number of mined patterns:%d\n",procRank, totalMinedPatterns);

#ifdef PAPI
			/* Stop the counters */
		retval = PAPI_stop(EventSet, values);
		if (retval != PAPI_OK) 
			handle_error(retval);
#endif
	consumedTime = get_elapsed_usec(start_time, end_time);
	reduce(world,consumedTime, maxConsumedTime, boost::parallel::maximum<long int>(),0);
	if(procRank == 0)
		printf("time consumed: %f seconds\n",maxConsumedTime/(float)CLOCKS_PER_SEC);
	/*std::cout<<"Mining done. Writing to log.."<<std::endl;

	std::ofstream output("output.log",std::fstream::out);
	std::set<Pattern>::const_iterator patternIter = patterns.begin();
	for(;patternIter!=patterns.end();patternIter++)
	{
		std::set<Item>::iterator itemIter=(patternIter->first).begin();
		for(;itemIter!=(patternIter->first).end();itemIter++)
		{
			output<<*itemIter<<" ";
		}
		output<<"("<<patternIter->second<<")"<<std::endl;
	}
	output.close();*/
#ifdef PAPI
	float  missRate = values[0]/(double)(values[1]);
	float  CPI = values[2]/(double)(values[3]);
	printf("L3 Miss Rate:%f CPI:%f Total number of instructions:%ld\n",missRate,CPI, values[3]);
#endif
    
    return EXIT_SUCCESS;
}


void ParallelFPTreeMining(void * inParam, int start, int end, void* outParam)
{
	TreesPendingMining& treesToBeMined = *(reinterpret_cast<TreesPendingMining*>(inParam));
    	std::set<Pattern>& patterns=*(reinterpret_cast<std::set<Pattern>*>(outParam));
	TreesPendingMining::iterator treeIter = treesToBeMined.begin()+start;	

	for(int i=start;i<end;i++,treeIter++)
	{
		std::set<Pattern> curr_item_patterns;
		TreesPendingMining _TTM;
		//std::set<Pattern> conditional_patterns=fptree_growth((*treeIter).first, spliceDepth+1,spliceDepth,(*treeIter).second, _TTM);
		fptree_growth(treeIter->first, spliceDepth+1,spliceDepth,treeIter->second, _TTM, patterns);
		//printf("%d: temp mined patterns %d\n",start,patterns.size());
		//delete treeIter->first;
		assert(_TTM.size()==0);
		/*// the next patterns are generated by adding the current item to each conditional pattern
		std::set<Pattern>::iterator cpIter = conditional_patterns.begin(); 
	    	for (cpIter = conditional_patterns.begin();cpIter !=conditional_patterns.end();cpIter++ ) 
		{
			Pattern new_pattern;
			new_pattern.first.insert(cpIter->first.begin(),cpIter->first.end());
			new_pattern.first.insert( (*treeIter).second);
			new_pattern.second = cpIter->second;
			curr_item_patterns.insert(new_pattern);
	    	}
	
	    	// join the patterns generated by the current item with all the other items of the fptree
	    	patterns.insert( curr_item_patterns.begin(), curr_item_patterns.end() );*/

	}
	
}

void ReadTransactionDB(std::vector<Transaction>& transactions, const char* fileName, int procRank)
{
	std::set<std::string> itemSet;
	int numTransactions=0;
	std::ifstream input(fileName, std::fstream::in);
	if(input.fail())
	{
		std::cout<<"File does not exist. exiting"<<std::endl;
		exit(0);
	}
 	if ( input.peek() != EOF ) 
    	{
		while(true) 
		{
			Transaction t;
			std::string transString, item;
			std::getline(input, transString);
			if(transString.size() == 0)
				break;
			std::stringstream strTokenizer(transString);
			while(strTokenizer)
			{
				std::getline(strTokenizer, item,' ');
				if(item.size() == 0)
					break;
				if((atoi(item.c_str())==-1) || (atoi(item.c_str())==-2))
				{	
					item.clear();
					continue;
				}

				itemSet.insert(item);
				//std::cout<<"item parsed "<<item<<" ";
				Item tmpItem(item);
				t.push_back(tmpItem); 
				item.clear();
			}
			//std::cout<<std::endl;
			transactions.push_back(t);
			numTransactions++;
		    	if(input.eof())
		    	{
				break;
			}
		}
	}
	input.close();
	if(procRank == 0)
		printf("%d items %d transactions\n",itemSet.size(),numTransactions);
}

void CommLineOpts::ParseCommLineOpts(int argc, char **argv, char c)
{
	int i=1;
	//std::cout<<"Command Line Options"<<std::endl;
	while(argc > 1)
	{
		const char* str = argv[i];
		const char* tmpStr = str;
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

bool CommLineOpts::Get(const std::string& paramName, void* val)
{
	bool flag = false;
	clopts::iterator iter=paramList.begin();
	while(iter != paramList.end())
	{
		if(!((iter->first).compare(paramName)))
		{
			if(!(paramName.compare("TRANSACTION_DB")))
			{
				const char * tmpName=(iter->second).c_str();
				memcpy(val,tmpName,(iter->second).length());
			}
			else
				*((unsigned *)val) = atoi((iter->second).c_str());	
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


#ifdef DEBUG
void PrintTreeStats(FPNode* node, std::map<char, int>& nodesAtHeight)
{
	std::pair<std::map<char,int>::iterator,bool> retVal = nodesAtHeight.insert(std::make_pair(node->height,1));
	if(!retVal.second)
		(retVal.first)->second += 1;
	std::vector<FPNode*>::iterator childIter=node->children.begin();
	for(;childIter!=node->children.end();childIter++)
	{
		PrintTreeStats(*childIter,nodesAtHeight);
	}
}
#endif

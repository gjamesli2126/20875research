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

#include "fptree.hpp"
#ifdef PAPI
#include"papi.h"
void handle_error(int retval)
{
	printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	exit(1);
}
#endif

#ifdef METRICS
int numberOfTraversals=0;
int numberOfVertices=0;
std::vector<FPNode*> subtrees;
class subtreeStats
{
public:
	long int footprint;
	int numnodes;
	subtreeStats(){footprint=0;numnodes=0;}
};
int splice_depth=4;
void getSubtreeStats(FPNode* ver, subtreeStats* stat);
void printLoadDistribution();
bool beginTrackingVertices=true;
#endif

unsigned spliceDepth; 
class CommLineOpts
{
private:	
	typedef std::vector<std::pair<std::string, std::string> > clopts;
	clopts paramList;
	void ParseCommLineOpts(int argc, const char **argv, char c='=');
public:
	CommLineOpts(int argc, const char **argv)
	{
		ParseCommLineOpts(argc,argv,'=');
	}

	bool Get(const std::string& paramName, void* val);
	void ParseCommaSeperatedValues(const char *str, std::vector<int>& retVal);
};


void ReadTransactionDB(std::vector<Transaction>& transactions, const char* fileName);
void ParallelFPTreeMining(void * inParam, int start, int end, void* outParam);

#ifdef DEBUG
void PrintTreeStats(FPNode* node, std::map<short int, int>& nodesAtVertices);
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



int main(int argc, char *argv[]) {
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
	
	if(argc < 2) 
	{
        	fprintf(stderr, "usage: fpgrowth [-t] <transaction_db_name> <min_support> <splice_depth(optional)>\n");
	        exit(1);
    	}

	struct timeval start_time, end_time;
	std::vector<Transaction> transactions;
    	unsigned minimum_support_treshold;
	char transactionDBName[255];
	memset(transactionDBName,0,255);


    	int i, j, c;
	int numThreads=1;
	spliceDepth = SPLICE_DEPTH;
	while((c = getopt(argc, argv, "t:")) != -1) {
        switch(c) {
            case 't':
                numThreads = atoi(optarg);
                if(numThreads <= 0) {
                    fprintf(stderr, "Error: invalid number of threads.\n");
                    exit(1);
                }
                i+=2;
                break;
            case '?':
                fprintf(stderr, "Error: unknown option.\n");
                exit(1);
                break;
                
            default:
                abort();
        }
    }
    
    for(i = optind; i < argc; i++) {
        switch(i - optind) {
	    case 0:
		strcpy(transactionDBName, argv[i]);
		break;
            case 1:
                minimum_support_treshold = atoi(argv[i]);
                break;
            case 2:
                spliceDepth = atoi(argv[i]);
                break;
	    default: 
		break;
        }
    }

	printf("Parameters used: SPLICE_DEPTH = %d MIN_SUPPORT=%d TRANSACTION_DB=%s NUM_THREADS=%d\n",spliceDepth, minimum_support_treshold,transactionDBName,numThreads);

	//minimum_support_treshold = atoi(argv[1]);
	ReadTransactionDB(transactions, transactionDBName);
	gettimeofday(&start_time, NULL);
	FPTree* fptree=new FPTree(transactions, minimum_support_treshold);
#ifdef METRICS
	printf("Number of vertices in prefix tree (With entire transaction DB) %d\n",numberOfVertices);
	numberOfVertices=0;
#endif
	gettimeofday(&end_time, NULL);
	long int consumedTime = get_elapsed_usec(start_time, end_time);
	printf("tree construction time: %f seconds\n",consumedTime/(float)CLOCKS_PER_SEC);
#ifdef DEBUG
	std::map<short int, int> nodesAtHeight;
	PrintTreeStats(fptree->root, nodesAtHeight);
	std::map<short int,int>::iterator mIter=nodesAtHeight.begin();
	for(;mIter!=nodesAtHeight.end();mIter++)
		printf("VerticesAtHeight %d : %d\n",mIter->first,mIter->second);
#endif

#ifdef METRICS
	printLoadDistribution();
#endif
	//printf("fptree header table size %d\n",fptree->header_table.size());
	printf("Constructing fptree done. Mining fptree...\n");
#ifdef PAPI
			retval = PAPI_start(EventSet);
			if (retval != PAPI_OK) handle_error(retval);
#endif
	gettimeofday(&start_time, NULL);
	TreesPendingMining treesToBeMined;
	Item nullPrefix;
    	//std::set<Pattern> patterns = fptree_growth( fptree, 0, spliceDepth, nullPrefix, treesToBeMined );
    	std::set<Pattern> patterns;
    	fptree_growth( fptree, 0, spliceDepth, nullPrefix, treesToBeMined, patterns);
	delete fptree;
	TreesPendingMining::iterator treeIter = treesToBeMined.begin();
	int noDuplicates=0;
	int numMinedPatterns=0;
	printf("mined patterns: %d, Pending trees to be mined:%d\n",patterns.size(), treesToBeMined.size());
	if((numThreads==1) || treesToBeMined.size() == 0)
	{
		ParallelFPTreeMining(&treesToBeMined, 0, treesToBeMined.size(), &patterns);
		numMinedPatterns=patterns.size();
		//printf("number of mined patterns:%d\n",patterns.size());
	}
	else
	{
		numMinedPatterns=patterns.size();
		pthread_t *threads = NULL;
		threads = new pthread_t[numThreads];
    		std::set<Pattern>* outParam = new std::set<Pattern>[numThreads];
		if(!threads) 
		{
			fprintf(stderr, "error: could not allocate threads.\n");
			exit(1);
		}
		struct targs *args = new struct targs[numThreads];
		if(!args) 
		{
			fprintf(stderr, "error: could not allocate thread args.\n");
			exit(1);
		}
		int *ranges = distribute_among(0, treesToBeMined.size(), numThreads);
		for(int i = 0; i < numThreads; i++) 
		{
			args[i].start = ranges[2 * i];
			args[i].end = ranges[2 * i + 1];
			args[i].func = ParallelFPTreeMining;
			args[i].funcIn = &treesToBeMined;
			args[i].funcOut = &(outParam[i]);
			if(pthread_create(&threads[i], NULL, thread_entry, &args[i]) != 0)
			{
				fprintf(stderr, "error: could not create thread.\n");
			}
		}
		delete [] ranges;

		for(int i = 0; i < numThreads; i++) 
		{
			pthread_join(threads[i], NULL);
			numMinedPatterns += outParam[i].size();
			//printf("%d: number of mined patterns:%d\n",i,outParam[i].size());
			//patterns.insert(outParam[i].begin(),outParam[i].end());
		}
		
		//assert(numMinedPatterns == patterns.size());
		delete [] threads;
		delete [] outParam;

	}
	gettimeofday(&end_time, NULL);
#ifdef PAPI
	/* Stop the counters */
	retval = PAPI_stop(EventSet, values);
	if (retval != PAPI_OK) 
		handle_error(retval);
#endif
	printf("Total number of mined patterns:%d\n",numMinedPatterns);
#ifdef METRICS
	printf("total number of traversals %d \n",numberOfTraversals);
	printf("total number of node visits %d\n",numberOfVertices);
#endif
	consumedTime = get_elapsed_usec(start_time, end_time);
	printf("time consumed: %f seconds\n",consumedTime/(float)CLOCKS_PER_SEC);
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

void ReadTransactionDB(std::vector<Transaction>& transactions, const char* fileName)
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
	printf("%d items %d transactions\n",itemSet.size(),numTransactions);
}

void CommLineOpts::ParseCommLineOpts(int argc, const char **argv, char c)
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
void PrintTreeStats(FPNode* node, std::map<short, int>& nodesAtHeight)
{
	std::pair<std::map<short,int>::iterator,bool> retVal = nodesAtHeight.insert(std::make_pair(node->height,1));
	if(!retVal.second)
		(retVal.first)->second += 1;
	std::vector<FPNode*>::iterator childIter=node->children.begin();
	for(;childIter!=node->children.end();childIter++)
	{
		PrintTreeStats(*childIter,nodesAtHeight);
	}
}
#endif

#ifdef METRICS
void printLoadDistribution()
{
	printf("num bottom subtrees %d\n",subtrees.size());
	std::vector<FPNode*>::iterator iter = subtrees.begin();
	for(;iter != subtrees.end();iter++)
	{
		long int num_vertices=0, footprint=0;
		subtreeStats stats;
		getSubtreeStats(*iter, &stats);
		printf("(%p) num_vertices %d footprint %ld\n",*iter, stats.numnodes, stats.footprint);
	}
}

void getSubtreeStats(FPNode* ver, subtreeStats* stats)
{
	stats->numnodes += 1;
	assert(ver != NULL);

	std::vector<FPNode*>::iterator childIter=ver->children.begin();
	for(;childIter!=ver->children.end();childIter++)
	{
		getSubtreeStats(*childIter,stats);
	}
}


#endif

#ifndef CLOPS_H
#define CLOPS_H

#define BLOCK_SIZE 1
#define BLOCK_INTAKE_LIMIT 1024
#define SUBTREE_HEIGHT 10
#define DGENTREE_VCOUNT 100

#ifdef MESSAGE_AGGREGATION
#define NUM_PIPELINE_BUFFERS 1 
#define DEFAULT_STAGE_NUM 999
#define DEFAULT_BUFFER_SIZE 256
#define PIPELINE_BUFFER_SIZE_LEVEL_1 64
#define PIPELINE_BUFFER_SIZE_LEVEL_2 1024
#define PIPELINE_BUFFER_SIZE_LEVEL_3 256
#define PIPELINE_BUFFER_SIZE(a,b) a##b
#define PIPELINE_STAGE_NUM(a, b, subtreeHeight) ((a) / (subtreeHeight+1))<=b?((a) / (subtreeHeight+1))+1:DEFAULT_STAGE_NUM;
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
	void PrintOpts();
	int blkSize;
	int subtreeHeight; 
	int blkIntakeLimit; 
	int dgentree_vcount;
#ifdef SPAD_2
	int numReplicatedSubtrees;
#endif
#ifdef MESSAGE_AGGREGATION
	int numBuffers; 
	std::vector<int> pipelineBufferSizes;
#endif
};
#endif

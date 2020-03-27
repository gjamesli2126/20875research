/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef SPIRIT_H
#define SPIRIT_H
#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<sys/shm.h>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include "Point.hpp"
#include "Messages.hpp"
#include "Optimizations.hpp"
using namespace boost;
using boost::graph::distributed::mpi_process_group;

#define BUILDSUBTREE_SAMEPROCESS 0
#define BUILDSUBTREE_MAXHEIGHT 2
#define STATUS_SUCCESS 0
#define STATUS_FAILURE -1
#define STATUS_PIPELINE_FULL -2
#define STATUS_NO_WORKITEMS -3
#define STATUS_TRAVERSE_INCOMPLETE -1
#define STATUS_TRAVERSE_COMPLETE 0

class SPIRITVisitor;
class WLManager;
typedef enum TVertexType{VTYPE_PSEUDOROOT, VTYPE_NORMAL}TVertexType;
typedef std::pair<int, long int> RepPLeaf; 

typedef struct PLeafPointBucket{
int subtreeOwner;
int numVerticesInSubtree;
}PLeafPointBucket;


class Vertex{
public:
#ifdef TRAVERSAL_PROFILE
	long int pointsVisited;
#endif
	int leftDesc;
	int rightDesc;
	int parentDesc;
	bool pseudoLeaf;
	bool pseudoRoot;
	Vertex* leftChild;
	Vertex* rightChild;
	Vertex* parent;
	long int label;
	char level;
	VertexData* vData;
	bool leaf;
	short int numPointsInCell;
	std::vector<char> cellNo;
	std::vector<Vertex*> pChild;
	std::vector<int> childDesc;
	Vertex():leftChild(0),rightChild(0),parent(0), pseudoRoot(false), pseudoLeaf(false), level(0),leaf(false),label(0), numPointsInCell(0)
	{
		vData = NULL;
		Vertex* tmp = NULL;
		for(int i=0;i<8;i++) pChild.push_back(tmp);
		for(int i=0;i<8;i++) childDesc.push_back(-1);
#ifdef TRAVERSAL_PROFILE
		pointsVisited=0;
#endif

	}
	virtual ~Vertex();
	void GetPLeafChildrenDetails(Vertex* nextNodeToVisit, MsgTraverse& msgTraverse);
};

class SPIRIT_PseudoRoot:public Vertex{
public:
	//std::vector<RepPLeaf> parents;
	std::map<int, long int> parents2;
	SPIRIT_PseudoRoot() {}
};

class SPIRIT
{
	public:
	static SPIRIT* GetInstance(Optimizations& opts, mpi_process_group& prg);
	void ReadTreeData(char* inputFile, long int numPoints, std::vector<Point*>& points, InputFileParser* parser);
	long int ReadTreeData_Oct(char* inputFile, long int numPointsRead, long int totalPoints, unsigned char* pointArr, InputFileParser* parser, int step);
	int ConstructBinaryTree(std::vector<Point*>& points, long int numPoints, InputFileParser* inputParser);
	//int ConstructBinaryTree(char* fileName, long int numPoints, InputFileParser* inputParser);
	float ConstructOctree(char* fileName, long int numPoints, InputFileParser* parser);
	int Traverse(SPIRITVisitor* v);
	void AllocateWorkspace(long int numPoints, InputFileParser* inputParser);
	void FreeWorkspace();
	void AggregateResults();
	void PrintResults();
	void PrintGraph();
	static void ResetInstance(){delete instance;instance=NULL;}

	void print_treetofile(FILE* fp);
	void print_preorder(Vertex* node, FILE* fp);
	private:
	static SPIRIT* instance;
	SPIRITVisitor* vis;
	WLManager* wlManager;
	Optimizations& opts;
	int numUpdatesRequired;
	std::set<int> readyToExitList;
	bool readyToExit;
	TType curTreeType;
	long int numTraversals;
	InputFileParser* parser;

	int subtreeHeight;
	int procCount;
	int procRank;
	int numProcs;
	mpi_process_group& pg;

	Vertex* rootNode; 
	boost::interprocess::mapped_region* sharedWorkspace;
	boost::interprocess::shared_memory_object* shm;
	unsigned char* address;
#ifdef MESSAGE_AGGREGATION
	std::map<long int, long int> aggrBuffer;
	bool readyToFlush;
#endif
#ifdef TRAVERSAL_PROFILE
	long int numPRootLeaves;
	std::map<int,long int> numLeavesAtHeight;
#endif
#ifdef STATISTICS
	long int traverseCount;
	long int bufferStage;
	long int pointsProcessed;
#endif

	SPIRIT(Optimizations& o,mpi_process_group& prg): opts(o), procCount(0), pg(prg)	
	{
#ifdef MESSAGE_AGGREGATION
		readyToFlush = false;
#endif
#ifdef TRAVERSAL_PROFILE
		numPRootLeaves = 0;
#endif
#ifdef STATISTICS
		bufferStage = 0;
		traverseCount = 0;
		pointsProcessed = 0;
#endif
		procRank = process_id(prg);
		numProcs = num_processes(prg);		
		numUpdatesRequired = 0;
		readyToExit = false;
		wlManager=NULL;
		rootNode = NULL;
		sharedWorkspace = NULL;
		shm=NULL;
		address=NULL;
		parser=NULL;
	}
	~SPIRIT()
	{
		if(shm)
		{
			shm->remove("SW1");
			delete sharedWorkspace;
			delete shm;
		}
		if(address)
			delete [] address;
	}

	Vertex* CreateVertex(TVertexType vType);
	void SendMessage(int processId, int messageId, void* message); 
	Vertex* CloneSubTree(std::vector<Point*>& points, Vertex* subtreeRoot, int from, int to, int depth);
	int SPIRIT_TraverseHelper(Vertex* node, Vertex* sib);
	int SPIRIT_TraverseHelper_SendBlocks();	
	int SPIRITHelper_HandleMessageTraverseBackward(Vertex* childNode, TBlockId curBlockId);
#ifdef MESSAGE_AGGREGATION
	int SPIRIT_TraverseHelper_CompressMessages(MsgTraverse& msgTraverse, int aggrBlockSize, MsgTraverse& msgTraverseRight, bool goingLeft=true);
	void SPIRIT_TraverseHelper_SendCompressedMessages();
	bool SPIRITHelper_HandleMessageTraverseBackward_Multiple(Vertex* parentNode, MsgTraverse& msg, int msgId);
#endif
	int SPIRITHelper_DeAggregateBlockAndSend(Vertex** pVertex, MsgTraverse& msgTraverse, bool loadBalanced);
	int SPIRITHelper_GetNextProcessId(long int pLeaf);
	void SPIRITHelper_CountSubtreeNodes(Vertex* ver, long int& count);
	void SPIRITHelper_CountSubtreeNodes_Oct(Vertex* ver, long int& count);
	void SPIRITHelper_ComputeBoundingBoxParams(const TPointVector& nbodies, Vec& center, float& dia);
	void SPIRITHelper_ComputeBoundingBoxParams(const unsigned char* nbodies, long int size, long int numPoints, Vec& center, float& dia);
	int SPIRITHelper_UpdatePLeaves_Oct(MsgUpdatePLeaves_Oct& msg);
	int SPIRITHelper_ComputeCofm(Vertex* node, bool explorePseudoLeafChildren=false);
	int SPIRITHelper_ComputeChildNumber(const Vec& cofm, const Vec& chCofm);
	int BuildSubTree_Oct(Vertex* subtreeRoot, int rOwner, Point* point, Vec& center, float dia, int depth, int DOR, bool clonePoint);
	void SPIRITHelper_ReplicateSubtrees(int numSubtreesToBeReplicated);
	void SPIRITHelper_ReplicateSubtrees_Oct(long int numSubtreesToBeReplicated);
	long int SPIRITHelper_SaveSubtree(Vertex* pRoot, TPointVector& ptv);
	void BlockCyclicDistribution(long int traversalsTBD, int subBlockSize);
#ifdef TRAVERSAL_PROFILE
	void SPIRITHelper_GetTotalPointsVisited(Vertex* ver, long int& count);
#endif

};



#endif

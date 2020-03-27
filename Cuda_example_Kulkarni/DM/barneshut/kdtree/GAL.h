/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef GAL_H
#define GAL_H

#define MERGE_HEIGHT1_TREES



/*Default values of parameters. They can also be defined at runtime.
 *====================================================*/
/* Message aggregation is a feature that buffers the blocks at pseudo leaves and sends them to the children of those pseudo roots only when the buffer fills up or when 
 * there are no more input blocks left to traverse. */
//#define MESSAGE_AGGREGATION //uncomment this only if the 'make pipelined' option is not defined in makefile.
#define NUM_PIPELINE_BUFFERS 1 
#define BLOCK_SIZE 4 
#define BLOCK_INTAKE_LIMIT 1024
#define SUBTREE_HEIGHT 1
/*====================================================*/
#define SIGNBIT_OFFSET 23
#define PID_OFFSET 24
/* Block Id (lowest 32 bits) is composed of <process_id>:<sign>:<blockId>
 * process_id = 5 bits == 32 processes supported.
 * sign = 1 bit = value of 1=> blockId is to be treated as negative number. This is always set to 1 when COMPOSE_BLOCK_ID is called. This is a differentiator when compared to normal block IDs.
 * blockId = 26 bits => maximum of 2^26 blocks are supported. so, assuming a unit sized block, KdTree can have atmost 2^26 leaves
 */
#define COMPOSE_BLOCK_ID(a,b) ((a<<PID_OFFSET) | (1<<SIGNBIT_OFFSET) | b)
#define PID_MASK 0xFF000000
#define SIGNBIT_MASK 0x00800000
#define BLOCKID_MASK 0x0007FFFFF
#define EXTRACT_BLOCK_ID(a) (a & BLOCKID_MASK) 
#define EXTRACT_PID(a) ((a & PID_MASK) >> PID_OFFSET) 
#define EXTRACT_SIGNBIT(a) ((a & SIGNBIT_MASK) >> SIGNBIT_OFFSET)


#ifdef MESSAGE_AGGREGATION
#define DEFAULT_STAGE_NUM 999
#define DEFAULT_BUFFER_SIZE 256
#define PIPELINE_BUFFER_SIZE_LEVEL_1 64
#define PIPELINE_BUFFER_SIZE_LEVEL_2 1024
#define PIPELINE_BUFFER_SIZE_LEVEL_3 256
#define PIPELINE_BUFFER_SIZE(a,b) a##b

/* 
 * Given a node's height, this macro returns DEFAULT_STAGE_NUM if the node is a pseudo leaf but is much down in the tree than the number of pipeline stages support.
 * Otherwise if the the node is a pseudo leaf and whose height is within the maximum possible height for a pseudo leaf to be buffering data, it returns the  stage of the pipeline buffer.
 * There is one buffer per stage of the pipeline. pipeline buffers are located at pseudo leaves at height = (multiples of SUBTREE_HEIGHT)  -1 ( -1 because height starts from 0)
 */

//#define PIPELINE_STAGE_NUM(a, b, subtreeHeight) ((subtreeHeight * b) % ((a)+1))==0?(b - (subtreeHeight * b / ((a)+1)) + 1): DEFAULT_STAGE_NUM;
#define PIPELINE_STAGE_NUM(a, b, subtreeHeight) ((a+1) % subtreeHeight)==0?( ((a+1)/subtreeHeight)<=b?((a+1)/subtreeHeight):DEFAULT_STAGE_NUM): DEFAULT_STAGE_NUM;
#endif

#define INVALID_BLOCK_ID 0
#define PARALLEL_BGL

#ifdef PARALLEL_BGL
#include "BGLKDTree.h"
#include <boost/graph/distributed/local_subgraph.hpp>
#include<boost/bind.hpp>
using namespace boost;
using boost::graph::distributed::mpi_process_group;
#define KDTREE 0

#ifdef STATISTICS2
extern int numBlocks;
extern double parOverhead;
extern long int workDone;
#endif

typedef enum TVertexType{VTYPE_PSEUDOROOT, VTYPE_NORMAL}TVertexType;
typedef std::vector<int> TIndices; //vector of index numbers (pointers) pointing to within the sub-block ( == blockStack entry)

class GAL_Vertex{
public:
#ifdef TRAVERSAL_PROFILE
	long int pointsVisited;
	double blockOccupancyFactor;
#endif
	BGLVertexdesc desc;
	int leftDesc;
	int rightDesc;
	int parentDesc;
	bool pseudoLeaf;
	bool pseudoRoot;
	CellParams cp;
	LeafParams* lp;

	GAL_Vertex* leftChild;
	GAL_Vertex* rightChild;
	GAL_Vertex* parent;
	long int label;
	short int level;
	bool isLeftChild;
#ifdef MESSAGE_AGGREGATION
	int numBlocksVisited;
#endif
	GAL_Vertex():leftDesc(-1), rightDesc(-1),leftChild(0),rightChild(0),parent(0), pseudoRoot(false), pseudoLeaf(false), level(0), label(0), lp(NULL)
	{
#ifdef TRAVERSAL_PROFILE
		pointsVisited=0;
		blockOccupancyFactor=0.;
#endif
#ifdef MESSAGE_AGGREGATION
		numBlocksVisited=0;
#endif
	}
};

/* Pseudo-Roots maintain a list of blocks that visit them. Only a subset of a block visited may proceed down the subtree rooted at a pseudo-root. Thus, the subset gets pushed onto block stack. 
 * However, when the subset completes its traversal, the entire block that visited earlier needs to be sent back to parent. Hence, the data structure 'SuperBlock' is defined. This structure contains
 * the entire block visited and a unique identifier- blockId. The same information can also be maintained on the block-stack. However, block stack management becomes cumbersome.
 **/ 
class SuperBlock
{
	public:
	TIndices block;
	TBlockId blockId;
	GAL_Vertex* nextNodeToVisit;
	char nextNodeProc;
	SuperBlock(TIndices& blk,TBlockId blkId, GAL_Vertex* ntv, int nnp):block(blk),blockId(blkId), nextNodeToVisit(ntv), nextNodeProc(nnp){}
};

typedef std::pair<int, long int> RepPLeaf; 
class GAL_PseudoRoot:public GAL_Vertex{
public:
#ifdef DYNAMIC_LB
	long int numPointsSeen;
#endif
	//std::vector<RepPLeaf> parents;
	std::map<int, long int> parents2;
	GAL_PseudoRoot()
	{
#ifdef DYNAMIC_LB
		numPointsSeen=0;
#endif
	}
	std::vector<SuperBlock> superBlocks; 
};

class GAL_Edge{
public:
#ifdef PARALLEL_BGL
	BGLEdgedesc desc;
	
#endif
};


//return messages
#define BUILDSUBTREE_FAILURE -1
#define BUILDSUBTREE_SENTOOB 1
#define BUILDSUBTREE_SAMEPROCESS 0
#define BUILDSUBTREE_MAXHEIGHT 2

#define STATUS_SUCCESS 0
#define STATUS_FAILURE -1
#define STATUS_PIPELINE_FULL -2
#define STATUS_NO_WORKITEMS -3
#define STATUS_SUSPEND -4
#define STATUS_TRAVERSE_INCOMPLETE -1
#define STATUS_TRAVERSE_COMPLETE 0
#define STATUS_MESSAGE_COMPRESSED 3

enum GAL_Message{
MESSAGE_POINTBUCKET,
MESSAGE_NEWBUCKETREQ,
MESSAGE_UPDATE_PLEAVES,
MESSAGE_READYTOEXIT,
MESSAGE_BUILDSUBTREE,
MESSAGE_DONESUBTREE,
MESSAGE_DONEKDTREE,
MESSAGE_CREATE_SUBTREE,
MESSAGE_TRAVERSE,
MESSAGE_TRAVERSE_BACKWARD,
#ifdef DYNAMIC_LB
MESSAGE_REPLICATE_SUBTREE,
#endif
#ifdef MESSAGE_AGGREGATION
MESSAGE_SENDCOMPRESSED
#endif
};

typedef struct MsgPointBucket{
friend class boost::serialization::access;
bool hasMoreData;
int pLeafId;
int depth;
bool isLeft;
long int pLeaf;
TPointVector ptv;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & hasMoreData & pLeafId & depth & isLeft & pLeaf & ptv;
  }

MsgPointBucket(int pseudoLeafId, bool leftTree, int treeLevel, long int pseudoLeaf):pLeafId(pseudoLeafId),isLeft(leftTree),depth(treeLevel),pLeaf(pseudoLeaf){hasMoreData=false;}
MsgPointBucket(){}
}MsgPointBucket;


typedef struct MsgBuildSubTree{
friend class boost::serialization::access;
#ifndef HYBRID_BUILD
long int from;
long int to;
#endif
long int pLeafId;
long int numPoints; //when ptv==0;
int depth;
bool isleft;
std::vector<std::pair<int,long int> >  pLeaf; //pid,pleaf pair
BGLVertexdesc subroot;
CellParams cpar;
TPointVector ptv;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
#ifdef HYBRID_BUILD
    ar & pLeafId & numPoints & depth & isleft & subroot & pLeaf & cpar & ptv;
#else
    ar & from & to & pLeafId & numPoints & depth & isleft & subroot & pLeaf & cpar & ptv;
#endif
  }
MsgBuildSubTree(bool leftTree, int treeLevel, BGLVertexdesc rootedAtVertex):pLeafId(-1), numPoints(0),isleft(leftTree),depth(treeLevel),subroot(rootedAtVertex)
{
#ifndef HYBRID_BUILD
from=to=0;
#endif
}
MsgBuildSubTree():numPoints(0), pLeafId(-1)
{
#ifndef HYBRID_BUILD
from=to=0;
#endif
}

}MsgBuildSubTree;

typedef struct MsgUpdateMinMax{
friend class boost::serialization::access;
int parent; //parent node that is to be updated also a pseudo leaf
long int pRoot; //pseudo root (GAL_Vertex* corresponding to child)
long int pLeaf; //pseudo leaf who is to be updated (GAL_Vertex* corresponding to parent)
bool isLeft;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & parent & pRoot & pLeaf & isLeft;
  }

MsgUpdateMinMax()
{
}
}MsgUpdateMinMax;


typedef struct MsgTerminate{
friend class boost::serialization::access;
BGLVertexdesc root; 
BGLVertexdesc leftChild; 
BGLVertexdesc rightChild;
long int pLeft;
long int pRight;

template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & root & leftChild & rightChild & pLeft & pRight;
  }
}MsgTerminate;

typedef struct MsgTraverse{
friend class boost::serialization::access;
TLocalDataVector l;
TBlockId blkStart;
long int pRoot; //pseudo root
long int pLeaf; //pseudo leaf who is to be updated (GAL_Vertex* corresponding to parent)
long int pSibling;
int siblingDesc;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & pRoot & pLeaf & l & blkStart & pSibling & siblingDesc;
    //ar & pRoot & pLeaf & l & blkStart;
  }
}MsgTraverse;

typedef struct MsgUpdatePLeaf{
friend class boost::serialization::access;
//BGLVertexdesc leafDesc;
int leftDesc;
int rightDesc;
long int pLeftChild;
long int pRightChild;
long int label;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & leftDesc & rightDesc & pLeftChild & pRightChild & label;
  }
}MsgUpdatePLeaf;

typedef struct MsgUpdatePLeaves{
friend class boost::serialization::access;
std::vector<MsgUpdatePLeaf> vPLeaves;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & vPLeaves;
  }
}MsgUpdatePLeaves;

typedef struct MsgUpdatePRoot{
friend class boost::serialization::access;
long int pRoot;
long int pLeaf;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & pRoot & pLeaf;
  }
}MsgUpdatePRoot;


typedef struct MsgReplicateSubtree{
friend class boost::serialization::access;
TBlockId blkStart;
std::vector<std::string> data;
TLocalDataVector l;
long int pLeaf;
bool isLeftChild;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & blkStart & data & l & pLeaf & isLeftChild;
  }
}MsgReplicateSubtree;

typedef BGLGraph GAL_Graph;
typedef std::pair<int, int> PBGL_oobMessage;
#endif

typedef struct BSEntry
{
	TIndices block;
	GAL_Vertex* nodeToVisit;
	BSEntry* next;
	BSEntry():nodeToVisit(NULL),next(NULL){}
	~BSEntry()
	{
		block.erase(block.begin(),block.end());
	}
#ifdef PERFCOUNTERS
	BSEntry* GetLast()
	{
		BSEntry* tmp=this;
		while(tmp->next)	
		{
			tmp=tmp->next;
		}
		return tmp;
	}
#endif
}BSEntry;

typedef struct BlockStack{
#ifdef PERFCOUNTERS
long int entryNode;
#endif
//std::pair<const GAL_Vertex*,long int> blkId;
TBlockId bStackId;
BSEntry* bStack;
void* parentBStack;
TBlockId parentBlockId;
int numFragments;
int nextNodePid;
#ifdef STATISTICS2
double startTime;
double parOverhead;
#endif
BlockStack()
{
	numFragments=0;
	next=NULL;
	prev=NULL;
	bStack=NULL;
	nextNodePid = -1;
	parentBStack = NULL;
#ifdef MESSAGE_AGGREGATION
	isBuffer = false;
#endif
#ifdef STATISTICS2
	parOverhead=0.;
#endif
}
~BlockStack()
{
	BSEntry*  tmp = bStack;
	while(tmp)
	{
		tmp = bStack->next;
		delete bStack;
		bStack = tmp;
	}
}
BlockStack *next, *prev;
#ifdef MESSAGE_AGGREGATION
std::vector<BlockStack*> leftBuffer, rightBuffer;
bool isBuffer;
#endif
}BlockStack;


typedef TPointVector::iterator TPointBlockIter; //an iterator to a vector of points
/*typedef std::list<BlockStack> BlockStackList;
typedef std::list<BlockStack>::iterator BlockStackListIter;*/
typedef BlockStack* BlockStackList;
typedef BlockStack* BlockStackListIter;


#ifdef MESSAGE_AGGREGATION
typedef std::vector<BlockStack*> TCBSet;
#endif


class GALVisitor{
public:
	virtual bool GALVisitor_VisitNode(const GAL_Vertex* node, GAL_Vertex* nextNodeToVisit, TIndices&, TIndices&, TBlockId& curBlkId) {};
	virtual int GALVisitor_GetNumberOfWorkItems(){}
	virtual void GALVisitor_UpdateBlockFromBlockStackTop(GAL_Vertex* searchNode = NULL){}
	virtual void GALVisitor_PushToBlockStackAndUpdate(MsgTraverse& msg){}
	virtual TBlockId GALVisitor_GetLocalData(TLocalDataVector& lData){}
	virtual int GALVisitor_SetLocalData(const TLocalDataVector& lData, TBlockId blkID, bool updateCurBlockStack=true){}
	virtual void GALVisitor_GetNextBlock(){}
	virtual int GALVisitor_FillPipeline(GAL_Vertex* startNode){}
	virtual void GALVisitor_SetAsCurrentBlockStack(const TBlockId& blkId){}
	virtual void GALVisitor_SetBlock(const TIndices& blk){}
	virtual TBlockId GALVisitor_DeleteFromSuperBlock(GAL_Vertex* pRoot, GAL_Vertex** nextNodeToVisit, int& nextNodeProc){}
	//virtual bool GALVisitor_IsSuperBlock(TBlockId blkId){}
	virtual void GALVisitor_AddToSuperBlock(GAL_Vertex* pRoot, TLocalDataVector& lData, TBlockId tmpBlockId, GAL_Vertex* nextNodeToVisit, int nextNodeProc){}
	virtual void GALVisitor_FlushTraversalReadyQ(){}
	virtual BlockStackListIter GALVisitor_CreateBlockStack(const TIndices& blk, const GAL_Vertex* rNode, GAL_Vertex* sibNode, int numExpUpdates, BlockStackListIter& curBStack){}
	virtual BlockStackListIter GALVisitor_GetCurrentBlockStack(){}
	virtual void GALVisitor_SetAsCurrentBlockStack2(BlockStackListIter curBStack){}
	virtual bool GALVisitor_IsLastFragment(GAL_Vertex* node, BlockStackListIter curBStack, TBlockId& parentBlkId){}	
	virtual int GALVisitor_RemoveBlockStack(){}
	virtual GAL_Vertex* GALVisitor_PopFromCurrentBlockStackAndUpdate(){}
	virtual void GALVisitor_UpdateCurrentBlockStackId(GAL_Vertex* pRoot){}
/*#ifdef PERFCOUNTERS
	virtual BlockStats GALVisitor_GetNumberOfBlocksInFlight(){}
#endif*/
	virtual int GALVisitor_GetWorkerId(long int workItemId){}
#ifdef MESSAGE_AGGREGATION
	virtual BlockStackListIter GALVisitor_CreateBufferStack(BlockStack* bufferStack, bool leftBuffer, int& curBufferSize){} 
	virtual TBlockId GALVisitor_GetBufferData(TLocalDataVector& lData, TLocalDataVector& rData){}
	virtual int GALVisitor_GetCompressedBlockIDs(TBlockId blockId, LocalData& lData, TCBSet& blkIdSet){}
	virtual bool GALVisitor_IsBufferedBlock(long int bStack){}
#endif

};

class GAL
{
	public:
	static GAL* GAL_GetInstance(mpi_process_group& prg);
	GAL_Graph& GAL_GetGraph(); 

	GAL_Vertex* GAL_CreateVertex(TVertexType vType);
	void GAL_SetVertexProperties(GAL_Vertex&, void* refnode, int vertexType);
	void* GAL_GetVertexProperties(GAL_Vertex&, int vertexType);

	void DetermineBoundingBox(Point* points,long int from, long int to, float* globalMin, float* globalMax);
	GAL_Vertex* Recurse(Point* points, GAL_Vertex* parent, long int from, long int to, int tmpHeight, int axis, bool isLeftChild, int& leafnodeId);
	long int DeterminePartitionElements(Point* points, long int from, long int to, float median, int axis);
	int GAL_ConstructKDTree(Point* points, int numPoints,bool hasMoreData, int subtreeHeight=SUBTREE_HEIGHT);

	void GAL_SendMessage(int processId, int messageId, void* message); 
	void GAL_PrintGraph();

	int GAL_Traverse(GALVisitor* v, int blkSize, int nBuffers, std::vector<int> pipelineBufferSizes);
	int GAL_TraverseHelper(GALVisitor* vis, GAL_Vertex* node, GAL_Vertex* sib);
	int GAL_TraverseHelper_SendBlocks(GALVisitor* vis);	
	int GALHelper_HandleMessageTraverseBackward(GAL_Vertex* childNode, GALVisitor* vis, TBlockId curBlockId);
	GAL_Vertex* GAL_GetRootNode();
#ifdef MESSAGE_AGGREGATION
	int GAL_TraverseHelper_CompressMessages(GALVisitor* vis, MsgTraverse& msgTraverse, int aggrBlockSize, MsgTraverse& msgTraverseRight, bool goingLeft=true);
	void GAL_TraverseHelper_SendCompressedMessages(GALVisitor* vis);
	int GALHelper_HandleMessageTraverseBackward_Multiple(GAL_Vertex* parentNode, GALVisitor* vis, TCBSet& blkIdSet);
#endif
	bool GALHelper_RemoveFromFragTable(GAL_Vertex* fragger, long int curBlockId, TBlockId& fragBlkId, bool forceErase = false);
	bool GALHelper_FindAndRemoveFromFragTable(GAL_Vertex* node, long int curBlkId, TBlockId& fragBlkId);	
	int GALHelper_DeAggregateBlockAndSend(GALVisitor* vis, GAL_Vertex** pVertex, MsgTraverse& msgTraverse, bool loadBalanced);
	int GALHelper_GetNextProcessId(long int pLeaf);
	void GAL_DeleteVertex(GAL_Vertex* node);
	void GALHelper_DeleteSubtree(GAL_Vertex* ver);
	void GALHelper_SaveSubtreeAsString(GAL_Vertex* subtreeRoot, std::vector<std::string>& subtreeStr, long int parentId, int childNum, bool requestNotPending);
	void GALHelper_SaveSubtreeInFile(GAL_Vertex* node, long int parentId, int childNum, std::ofstream& fp);
	void GALHelper_ReplicateSubtreeFromString(MsgReplicateSubtree& msgCloneSubtree, GAL_Vertex** pRoot, GAL_Vertex* pLeaf, int pLeafOwner);
	void GALHelper_CreateSubtreeFromString(MsgReplicateSubtree& msgCloneSubtree, GAL_Vertex** pRoot, GAL_Vertex* pLeaf, int pLeafOwner);
	void GALHelper_CreateSubtreeFromFile(GAL_Vertex** pRoot, GAL_Vertex* pLeaf, int pLeafOwner);
	void GALHelper_ReplicateTopSubtree();

	void compute_cell_params(Point* points, long int lb, long int ub, Vec& cofm, float& mass, float& radius);
#ifdef DYNAMIC_LB
	void GALHelper_ReplicateSubtrees();
	int GALHelper_GetRandomSubtreeAndBroadcast();
	long GALHelper_GetRandom(int size);
	bool GALHelper_IsBottleneckSubtree(int label, int** bneckList);
#endif
	void GALHelper_CountSubtreeNodes(GAL_Vertex* ver, long int& count);
	void GALHelper_GetBlockOccupancyFactor(GAL_Vertex* ver, double& count);
#ifdef TRAVERSAL_PROFILE
	long int numLinks;
	long int numPRootLeaves;
	std::map<int,long int> numLeavesAtHeight;
#endif
#ifdef STATISTICS
	long int traverseCount;
	long int *bufferStage;
	long int pointsProcessed;
#endif

	void print_treetofile(FILE* fp);
	void print_preorder(GAL_Vertex* node, FILE* fp);
	private:
	int numUpdatesRequired;
	std::set<int> readyToExitList;
	bool readyToExit;

	int subtreeHeight;
	int blockSize;
	int procCount;
	int procRank;
	int numProcs;
	int vertextype;
	mpi_process_group& pg;
	GAL(mpi_process_group& prg): procCount(1),pg(prg)	
	{
#ifdef PARALLEL_BGL
		g = new BGLGraph();//if distribution changes, constructor can be passed the numvertices argument 
#endif
		vertextype = KDTREE;
		curAggrBlkId = 1;
#ifdef MESSAGE_AGGREGATION
		readyToFlush = false;
		numPipelineBuffers=NUM_PIPELINE_BUFFERS; 
		pipelineBufferSize = NULL;
#endif
#ifdef TRAVERSAL_PROFILE
		numLinks = 0;
		numPRootLeaves = 0;
#endif
#ifdef STATISTICS
		bufferStage = NULL;
		traverseCount = 0;
		pointsProcessed = 0;
#endif
		blockSize = BLOCK_SIZE;
		procRank = process_id(prg);
		numProcs = num_processes(prg);		
		subtreeHeight = SUBTREE_HEIGHT;
		numUpdatesRequired = 0;
		readyToExit = false;
		rootNode=NULL;
	}
	int GAL_BuildSubTree(Point*, GAL_Vertex* subtreeRoot,int from, int to, int height, int DOR, bool isLeft, CellParams& cpar);
	int GAL_Aux_UpdatePLeaves(MsgUpdatePLeaves& msg);
	bool GAL_Aux_IsReadyToExit(){return (numUpdatesRequired == 0)?true:false;}
	void GALHelper_WriteVertexToFile(std::vector<MsgUpdatePLeaf>& msg);
	void GALHelper_CreateVerticesFromFile(MsgUpdatePLeaves& msgUpdatePLeaves);
	static GAL* instance;
	GAL_Graph* g;
	GAL_Vertex* rootNode; 
	int curAggrBlkId;
#ifdef MESSAGE_AGGREGATION
	std::map<long int, long int> aggrBuffer;
	//std::list<CompressedBlkObj> deFragTable;
	bool readyToFlush;
	int numPipelineBuffers;
	int *pipelineBufferSize;
#endif
#ifdef PERFCOUNTERS
	uint64_t totTime;
	uint64_t idleTime;
	uint64_t compTime;
	uint64_t leafStageTime;
#endif


};


#endif

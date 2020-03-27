/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef GAL_H
#define GAL_H

#define MERGE_HEIGHT1_TREES
//#define STATISTICS
//#define OPT


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
 * process_id = 8 bits == 32 processes supported.
 * sign = 1 bit = value of 1=> blockId is to be treated as negative number. This is always set to 1 when COMPOSE_BLOCK_ID is called. This is a differentiator when compared to normal block IDs.
 * blockId = 23 bits => maximum of 2^23 blocks are supported. so, assuming a unit sized block, KdTree can have atmost 2^26 leaves
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
#define INVALID_BLOCK_ID -1
#define PARALLEL_BGL

#ifdef PARALLEL_BGL
#include "BGLKDTree.h"
#include <boost/graph/distributed/local_subgraph.hpp>
#include<boost/bind.hpp>
using namespace boost;
#define VPTREE 0


typedef enum TVertexType{VTYPE_PSEUDOROOT, VTYPE_NORMAL}TVertexType;
typedef std::vector<int> TIndices; //vector of index numbers (pointers) pointing to within the sub-block ( == blockStack entry)
/*typedef struct FragBlock
{
	TBlockId inBlkId;
	std::set<long int> uniqBlkIds;
}FragTableVal;

typedef struct FragTableElm
{
	std::list<FragBlock> fragBlocks;	
}FragTableKey;
typedef std::list<FragTableElm>::iterator FragTableIter;*/


class GAL_Vertex{
public:
#ifdef TRAVERSAL_PROFILE
	long int pointsVisited;
	double blockOccupancyFactor;
#endif
#ifdef PARALLEL_BGL
	BGLVertexdesc desc;
	int leftDesc;
	int rightDesc;
	int parentDesc;
	bool pseudoLeaf;
	bool pseudoRoot;
#endif
	GAL_Vertex* leftChild;
	GAL_Vertex* rightChild;
	GAL_Vertex* parent;
	long int label;
	short int level;
	bool isLeftChild;
	float threshold;
	Point point;
	bool leaf;
	GAL_Vertex():leftChild(0),rightChild(0),parent(0), pseudoRoot(false), pseudoLeaf(false), level(0),leaf(false),threshold(0.), label(0)
	{
#ifdef TRAVERSAL_PROFILE
		pointsVisited=0;
		blockOccupancyFactor=0.;
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
	std::vector<RepPLeaf> parents;
	float parentThreshold;
	float parentCoord[DIMENSION];
	GAL_PseudoRoot(): parentThreshold(0.){}
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
#define STATUS_TRAVERSE_INCOMPLETE -1
#define STATUS_TRAVERSE_COMPLETE 0
#define STATUS_MESSAGE_COMPRESSED 3

enum GAL_Message{
MESSAGE_UPDATE_PLEAVES,
MESSAGE_UPDATE_PROOT,
MESSAGE_READYTOEXIT,
MESSAGE_BUILDSUBTREE,
MESSAGE_DONESUBTREE,
MESSAGE_DONEVPTREE,
MESSAGE_TRAVERSE,
MESSAGE_TRAVERSE_BACKWARD,
MESSAGE_SENDCOMPRESSED
};

typedef struct MsgBuildSubTree{
friend class boost::serialization::access;
int from;
int to;
int depth;
bool isleft;
long int pLeaf;
BGLVertexdesc subroot;
TPointVector ptv;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & from & to & depth & isleft & subroot &pLeaf & ptv;
  }

MsgBuildSubTree(int fromPoint, int toPoint, bool leftTree, int treeLevel, BGLVertexdesc rootedAtVertex):from(fromPoint),to(toPoint),isleft(leftTree),depth(treeLevel),subroot(rootedAtVertex){}
MsgBuildSubTree(){}

}MsgBuildSubTree;


typedef struct MsgUpdateMinMax{
friend class boost::serialization::access;
int parent; //parent node that is to be updated also a pseudo leaf
int child;	//the child node also a pseudo root
long int pRoot; //pseudo root (GAL_Vertex* corresponding to child)
long int pLeaf; //pseudo leaf who is to be updated (GAL_Vertex* corresponding to parent)
bool isLeft;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & parent & pRoot & pLeaf & isLeft & child;
  }
}MsgUpdateMinMax;


typedef struct MsgTerminate{
friend class boost::serialization::access;
BGLVertexdesc root; 
int leftChild; 
int rightChild;
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
float pLeafThreshold;
float pLeafCoords[DIMENSION];
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & pRoot & pLeaf & l & blkStart & pSibling & siblingDesc & pLeafThreshold;
    for(int i=0;i<DIMENSION;i++)
    {
	ar & pLeafCoords[i];
    }
    //ar & pRoot & pLeaf & l & blkStart;
  }
}MsgTraverse;

typedef struct MsgUpdatePLeaf{
friend class boost::serialization::access;
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

typedef BGLGraph GAL_Graph;
typedef std::pair<int, int> PBGL_oobMessage;
#endif

#ifdef PERFCOUNTERS
typedef struct BlockStats
{
	int numUniqBlocks;
	int numFinishedBlocks;
}BlockStats;
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
}BSEntry;

typedef struct BlockStack{
//std::pair<const GAL_Vertex*,long int> blkId;
TBlockId bStackId;
BSEntry* bStack;
void* parentBStack;
TBlockId parentBlockId;
int numFragments;
int nextNodePid;
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
	virtual void GALVisitor_FlushTraversalReadyQ(){}
	virtual BlockStackListIter GALVisitor_CreateBlockStack(const TIndices& blk, const GAL_Vertex* rNode, GAL_Vertex* sibNode, int numExpUpdates, BlockStackListIter& curBStack){}
	virtual BlockStackListIter GALVisitor_GetCurrentBlockStack(){}
	virtual void GALVisitor_SetAsCurrentBlockStack2(BlockStackListIter curBStack){}
	virtual bool GALVisitor_IsLastFragment(GAL_Vertex* node, BlockStackListIter curBStack, TBlockId& parentBlkId){}	
	virtual int GALVisitor_RemoveBlockStack(){}
	virtual GAL_Vertex* GALVisitor_PopFromCurrentBlockStackAndUpdate(){}
	virtual void GALVisitor_UpdateCurrentBlockStackId(GAL_Vertex* pRoot){}
#ifdef PERFCOUNTERS
	virtual BlockStats GALVisitor_GetNumberOfBlocksInFlight(){}
	virtual TBlockId GALVisitor_GetCurrentBlockId(){}
	virtual long int GALVisitor_DetermineOrigBlkId(LocalData& ld){}
#endif
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

	int GAL_ConstructVPTree(TPointVector& points, int subtreeHeight = SUBTREE_HEIGHT);

	void GAL_SendMessage(int processId, int messageId, void* message); 

	GAL_Edge GAL_CreateEdge(GAL_Vertex& source, GAL_Vertex& target);
	void GAL_SetEdgeProperties(GAL_Edge&, void* refedge, int vertexType);
	void* GAL_GetEdgeProperties(GAL_Edge&, int vertexType);

	GAL_Vertex* GAL_GetStartVertex(); 
	void GAL_PrintGraph();

	void GAL_ComputeCorrelation(TPointVector& ptv, float radius, int pid);
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
	void GALHelper_GetNodeData(GAL_Vertex* node, MsgTraverse& msg, bool bkwdDirection=true);
	void GALHelper_SetNodeData(GAL_Vertex* node, const MsgTraverse& msg);
	int GALHelper_GetNextProcessId(long int pLeaf);
	void GALHelper_CountSubtreeNodes(GAL_Vertex* ver, long int& count);
	void GALHelper_GetBlockOccupancyFactor(GAL_Vertex* ver, double& count);
#ifdef PERFCOUNTERS
	GAL_Vertex* GALHelper_GetAncestorPseudoRoot(GAL_Vertex* node);
#endif
	long int numVertices;
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
	private:
	int numUpdatesRequired;
	long int replicatedVertexCounter;
	std::set<int> readyToExitList;
	bool readyToExit;

	int subtreeHeight;
	int blockSize;
	int procCount;
	int procRank;
	int numProcs;
	int vertextype;
	mpi_process_group& pg;
	GAL(mpi_process_group& prg): procCount(1), pg(prg)	
	{
#ifdef PARALLEL_BGL
			g = new BGLGraph();//if distribution changes, constructor can be passed the numvertices argument 
#endif
		vertextype = VPTREE;
		curAggrBlkId = 1;
#ifdef MESSAGE_AGGREGATION
		readyToFlush = false;
		numPipelineBuffers=NUM_PIPELINE_BUFFERS; //by default,  a two stage pipeline is formed.
		pipelineBufferSize = NULL;
#endif
		numVertices = 0;
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
		replicatedVertexCounter = 0;
		readyToExit = false;
	}
	int GAL_BuildSubTree(TPointVector& points, GAL_Vertex* subtreeRoot,int from, int to, int depth, int DOR, bool isLeft);
	int GAL_Aux_UpdatePLeaves(MsgUpdatePLeaves& msg, int pid);
	bool GAL_Aux_IsReadyToExit(){return (numUpdatesRequired == 0)?true:false;}
	void GALHelper_CountReplicatedVertices(GAL_Vertex* subtreeRoot);

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
#endif


};



float mydistance(const Point& a, const Point& b);
#endif

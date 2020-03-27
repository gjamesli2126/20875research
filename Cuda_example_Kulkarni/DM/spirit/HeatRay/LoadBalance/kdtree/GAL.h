#ifndef GAL_H
#define GAL_H

#define LOAD_BALANCE
#define MERGE_DEGEN_TREES
#define MERGE_HEIGHT1_TREES
//#define STATISTICS


/*Default values of parameters. They can also be defined at runtime.
 *====================================================*/
/* Message aggregation is a feature that buffers the blocks at pseudo leaves and sends them to the children of those pseudo roots only when the buffer fills up or when 
 * there are no more input blocks left to traverse. */
//#define MESSAGE_AGGREGATION //uncomment this only if the 'make pipelined' option is not defined in makefile.
/*#define NUM_PIPELINE_BUFFERS 1 
#define BLOCK_SIZE 4 
#define BLOCK_INTAKE_LIMIT 1024
#define SUBTREE_HEIGHT 1*/
/*====================================================*/
#define SIGNBIT_OFFSET 23
#define PID_OFFSET 24
#define COMPOSE_BLOCK_ID(a,b) ((a<<PID_OFFSET) | (1<<SIGNBIT_OFFSET) | b)
/*#define DEFAULT_STAGE_NUM 999
#define DEFAULT_BUFFER_SIZE 256
#define PIPELINE_BUFFER_SIZE_LEVEL_1 64
#define PIPELINE_BUFFER_SIZE_LEVEL_2 1024
#define PIPELINE_BUFFER_SIZE_LEVEL_3 256
#define PIPELINE_BUFFER_SIZE(a,b) a##b*/

/* 
 * Given a node's height, this macro returns DEFAULT_STAGE_NUM if the node is a pseudo leaf but is much down in the tree than the number of pipeline stages support.
 * Otherwise if the the node is a pseudo leaf and whose height is within the maximum possible height for a pseudo leaf to be buffering data, it returns the  stage of the pipeline buffer.
 * There is one buffer per stage of the pipeline. pipeline buffers are located at pseudo leaves at height = (multiples of SUBTREE_HEIGHT)  -1 ( -1 because height starts from 0)
 */

//#define PIPELINE_STAGE_NUM(a, b, subtreeHeight) ((a+1) % subtreeHeight)==0?( ((a+1)/subtreeHeight)<=b?((a+1)/subtreeHeight):DEFAULT_STAGE_NUM): DEFAULT_STAGE_NUM;

/* Block Id (lowest 32 bits) is composed of <process_id>:<sign>:<blockId>
 * process_id = 5 bits == 32 processes supported.
 * sign = 1 bit = value of 1=> blockId is to be treated as negative number. This is always set to 1 when COMPOSE_BLOCK_ID is called. This is a differentiator when compared to normal block IDs.
 * blockId = 26 bits => maximum of 2^26 blocks are supported. so, assuming a unit sized block, KdTree can have atmost 2^26 leaves
 */

#define PID_MASK 0xFF000000
#define SIGNBIT_MASK 0x00800000
#define BLOCKID_MASK 0x0007FFFFF
#define EXTRACT_BLOCK_ID(a) (a & BLOCKID_MASK) 
#define EXTRACT_PID(a) ((a & PID_MASK) >> PID_OFFSET) 
#define EXTRACT_SIGNBIT(a) ((a & SIGNBIT_MASK) >> SIGNBIT_OFFSET)
#define INVALID_BLOCK_ID -1
#define PARALLEL_BGL
#define KDTREE 0

#ifdef PARALLEL_BGL
#include<stdio.h>
#include <boost/serialization/vector.hpp>
#include "shared/IntersectInfo.h"
#include "kdtree/TriangleKDTreeNode.h"
#include "util/clops.h"
#include "shared/Photon.h"

using namespace boost;
using boost::graph::distributed::mpi_process_group;

typedef adjacency_list<listS, distributedS<mpi_process_group,vecS>, undirectedS> BGLGraph;
typedef graph_traits<BGLGraph>::vertex_descriptor BGLVertexdesc;
typedef std::vector<gfx::Triangle> TriangleVector;
typedef std::pair<int, long int> TBlockId;


typedef enum TVertexType{VTYPE_PSEUDOROOT, VTYPE_NORMAL}TVertexType;
typedef std::vector<int> TIndices; //vector of index numbers (pointers) pointing to within the sub-block ( == blockStack entry)
typedef struct LocalData{
	friend class boost::serialization::access;
	int index; 
#ifdef TRACK_TRAVERSALS
	long int nodesTraversed;	//ray specific
#endif
	bool intersects;		//ray specific
	IntersectInfo info;
#ifdef STATISTICS2
	long int numStagesExecuted;
#endif
	template<typename Archiver>
	void serialize(Archiver& ar, const unsigned int) 
	{
#ifdef TRACK_TRAVERSALS
#ifdef STATISTICS2
	    ar & index & nodesTraversed & intersects & info & numStagesExecuted;
#else
	    ar & index & nodesTraversed & intersects & info;
#endif
#else
	    ar & index & intersects & info;
#endif
  	}

}LocalData;
typedef std::vector<LocalData> TLocalDataVector; 

class GAL_Vertex: public TriangleKDTreeNode{
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
	long int id;
#endif
	GAL_Vertex* leftChild;
	GAL_Vertex* rightChild;
	GAL_Vertex* parent;
	char level;
	bool isLeftChild;
	int uCount;
	GAL_Vertex():leftChild(0),rightChild(0),parent(0), pseudoRoot(false), pseudoLeaf(false), level(0),uCount(0)
	{
#ifdef TRAVERSAL_PROFILE
		pointsVisited=0;
		blockOccupancyFactor=0.;
#endif
	}
	bool isLeaf(void) const
	{
		return (leftChild == NULL && rightChild == NULL);
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

#ifdef LOAD_BALANCE
typedef std::pair<int, long int> RepPLeaf; 
#endif

class GAL_PseudoRoot:public GAL_Vertex{
public:
#ifdef LOAD_BALANCE
	std::vector<RepPLeaf> parents;
#endif
#ifdef SPAD_2
	long int pLeafLabel;
#endif
	GAL_PseudoRoot(){}
	std::vector<SuperBlock> superBlocks; 
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
#ifdef LOAD_BALANCE
MESSAGE_UPDATE_PLEAVES,
MESSAGE_UPDATE_PROOT,
MESSAGE_REPLICATE_PLEAF,
MESSAGE_READYTOEXIT,
#endif
MESSAGE_BUILDSUBTREE,
MESSAGE_DONESUBTREE,
MESSAGE_DONEKDTREE,
MESSAGE_TRAVERSE,
MESSAGE_TRAVERSE_BACKWARD,
#ifdef MESSAGE_AGGREGATION
MESSAGE_SENDCOMPRESSED,
#endif
#ifdef SPAD_2
MESSAGE_REPLICATE_REQ,
MESSAGE_REPLICATE_SUBTREE,
#endif
MESSAGE_DATA
};

#ifdef SPAD_2
typedef struct SubtreeHeader{
long int pRoot;
long int pLeaf;
char childNum;
int pRootDesc;
SubtreeHeader(long int _pRoot, char _cNum, int _pRootDesc):pRoot(_pRoot),childNum(_cNum),pRootDesc(_pRootDesc){}
SubtreeHeader();
}SubtreeHeader;

typedef struct MsgReplicateReq{
friend class boost::serialization::access;
long int pRoot;
long int pLeaf;
long int pLeafLabel;
char childNum;
int numSubtreesReplicated;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & childNum & pRoot & pLeaf & pLeafLabel & numSubtreesReplicated;
  }
MsgReplicateReq(){}
}MsgReplicateReq;

typedef struct MsgReplicateSubtree{
friend class boost::serialization::access;
std::vector<std::string> data;
long int pLeaf;
long int pLeafLabel;
char childNum;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & pLeaf & childNum & pLeafLabel & data;
  }
}MsgReplicateSubtree;

#endif

typedef struct MsgBuildSubTree{
friend class boost::serialization::access;
long int pLeafLabel; 
long int runningId;
int depth;
bool isleft;
BGLVertexdesc subroot;
long int pLeaf;
TIndices ptv;
Box box;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & runningId & depth & isleft & subroot & pLeaf & box & ptv & pLeafLabel;
  }

MsgBuildSubTree(bool leftTree, int treeLevel, BGLVertexdesc rootedAtVertex):isleft(leftTree),depth(treeLevel),subroot(rootedAtVertex){}
MsgBuildSubTree(){}

}MsgBuildSubTree;


typedef struct MsgUpdateMinMax{
friend class boost::serialization::access;
long int runningId;
int child;	//the child node also a pseudo root
long int pRoot; //pseudo root (GAL_Vertex* corresponding to child)
long int pLeaf; //pseudo leaf who is to be updated (GAL_Vertex* corresponding to parent)
bool isLeft;
#ifdef MERGE_HEIGHT1_TREES
TIndices ptv;
Box box;
#endif
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
#ifdef MERGE_HEIGHT1_TREES
    ar & runningId & pRoot & pLeaf & isLeft & child & ptv & box;
#else
    ar & runningId & pRoot & pLeaf & isLeft & child;
#endif
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
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & pRoot & pLeaf & l & blkStart & pSibling & siblingDesc;
    //ar & pRoot & pLeaf & l & blkStart;
  }
}MsgTraverse;


#ifdef LOAD_BALANCE
/* Message for creating an edge between pseudo leaf having a height = SUBTREE_HEIGHT and first level pseudo root that is not replicated */
typedef struct MsgUpdatePLeaf{
friend class boost::serialization::access;
BGLVertexdesc leafDesc;
int leftDesc;
int rightDesc;
long int pLeftChild;
long int pRightChild;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & leafDesc & leftDesc & rightDesc & pLeftChild & pRightChild;
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
int leafDesc;	
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & pRoot & pLeaf & leafDesc;
  }
}MsgUpdatePRoot;

typedef struct MsgReplicatedVertex{
friend class boost::serialization::access;
long int vert; 
BGLVertexdesc desc;
template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int /*version*/) {
    ar & vert & desc;
  }
}MsgReplicatedVertex;

#endif

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


/*typedef std::list<BlockStack> BlockStackList;
typedef std::list<BlockStack>::iterator BlockStackListIter;*/
typedef BlockStack* BlockStackList;
typedef BlockStack* BlockStackListIter;


#ifdef MESSAGE_AGGREGATION
typedef std::vector<BlockStack*> TCBSet;
#endif


class GALVisitor{
public:
	virtual bool GALVisitor_VisitNode(GAL_Vertex* node, GAL_Vertex* nextNodeToVisit, TIndices&, TIndices&, TBlockId& curBlkId) {};
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
	virtual int Traverse(){}
#ifdef PERFCOUNTERS
	virtual BlockStats GALVisitor_GetNumberOfBlocksInFlight(){}
#endif
#ifdef LOAD_BALANCE
	virtual int GALVisitor_GetWorkerId(long int workItemId){}
	virtual void GALVisitor_AssignWorkItems(long int startIndex, long int endIndex){}
	virtual TBlockId GALVisitor_GetCurrentBlockId(){}
#endif
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
	static GAL* GAL_GetInstance(mpi_process_group& prg, CommLineOpts* opts);
	GAL_Vertex* GAL_GetRootNode();
	GAL_Vertex* GAL_CreateVertex(TVertexType vType);
	void GAL_DeleteVertex(GAL_Vertex* node);
	void GAL_SendMessage(int processId, int messageId, void* message); 
	int GAL_ConstructKDTree(TriangleVector& points);
	void GAL_PrintGraph();

	int GAL_Traverse(GALVisitor* v);
	int GAL_TraverseHelper(GALVisitor* vis, GAL_Vertex* node, GAL_Vertex* sib);
	int GAL_TraverseHelper_SendBlocks(GALVisitor* vis);	
	int GALHelper_HandleMessageTraverseBackward(GAL_Vertex* childNode, GALVisitor* vis, TBlockId curBlockId);
#ifdef MESSAGE_AGGREGATION
	int GAL_TraverseHelper_CompressMessages(GALVisitor* vis, MsgTraverse& msgTraverse, int aggrBlockSize, MsgTraverse& msgTraverseRight, bool goingLeft=true);
	void GAL_TraverseHelper_SendCompressedMessages(GALVisitor* vis);
	int GALHelper_HandleMessageTraverseBackward_Multiple(GAL_Vertex* parentNode, GALVisitor* vis, TCBSet& blkIdSet);
	int GetOpt_PipelineBufferSizes(int stageNum);
	int GetOpt_NumPipelineBuffers();
#endif
	bool GALHelper_RemoveFromFragTable(GAL_Vertex* fragger, long int curBlockId, TBlockId& fragBlkId, bool forceErase = false);
	bool GALHelper_FindAndRemoveFromFragTable(GAL_Vertex* node, long int curBlkId, TBlockId& fragBlkId);	
	int GALHelper_DeAggregateBlockAndSend(GALVisitor* vis, GAL_Vertex** pVertex, MsgTraverse& msgTraverse, bool loadBalanced);
	int GetOpt_BlockSize();
	int GetOpt_BlockIntakeLimit();
	int GetOpt_SubtreeHeight();
#ifdef SPAD_2
	int GetOpt_NumReplicatedSubtrees();
#endif
#ifdef MERGE_DEGEN_TREES
	int GetOpt_DegentreeVCount();
#endif
	void SetOpt_SubtreeHeight(int st);
	int GetOpt_ProcRank(){return procRank;}
	int GetOpt_NumProcs(){return numProcs;}
	void GAL_Synchronize();
	template<typename T>
	void GAL_Test(T& obj);
	void GAL_AggregateResults(uint64_t localSum, double localTime, uint64_t& totalSum, double& totalTime);
	void GAL_AggregateResults_Generic(long int localSum, long int& totalSum);
#ifdef LOAD_BALANCE
	void AllGather(std::vector<Photon>& localPhotons, std::vector<Photon>& globalPhotons);
	//void GALHelper_CountReplicatedVertices(GAL_Vertex* subtreeRoot);
#endif
#ifdef SPAD_2
	void GALHelper_ReplicateSubtrees();
	int GALHelper_GetRandomSubtreeAndBroadcast();
	void GALHelper_ReadBottleneckDetails(std::ifstream& input, int numBottlenecks, std::vector<SubtreeHeader>& bottlenecks);
	int GALHelper_GetBottleneckSubtreesAndBroadcast(int numBottlenecks, char* bneckfile);
	bool GALHelper_IsBottleneck(std::vector<SubtreeHeader>& bneck, GAL_Vertex* pRoot);
#endif

#ifdef PERFCOUNTERS
	void GAL_StartComputeTimer();
	void GAL_StopComputeTimer();
#endif
	void GALHelper_CountSubtreeNodes(GAL_Vertex* ver, long int& count, bool profilingData=false, bool isRootSubtree=false);
	void GALHelper_GetBlockOccupancyFactor(GAL_Vertex* ver, double& count);
	void GALHelper_DeleteSubtree(GAL_Vertex* ver);
	bool GALHelper_IsStage(long int pRoot);
	//long int numVertices;
#ifdef TRAVERSAL_PROFILE
	long int numPRootLeaves;
	std::map<int,long int> numLeavesAtHeight;
#endif
#ifdef STATISTICS
	long int traverseCount;
	long int *bufferStage;
	long int pointsProcessed;
#endif
	mpi_process_group& pg;
	int numProcs;
	int procRank;

	void print_treetofile(std::ofstream& fp);
	void print_preorder(GAL_Vertex* node, std::ofstream& fp);
	private:
#ifdef LOAD_BALANCE
	//long int replicatedVertexCounter;
	int numUpdatesRequired;
	std::set<int> readyToExitList;
	bool readyToExit;
#endif
	long int* nodeCountTable;
	int procCount;
	int vertextype;
	CommLineOpts* clopts;
	GAL(mpi_process_group& prg, CommLineOpts* opts): procCount(1), pg(prg),clopts(opts)	
	{
#ifdef PARALLEL_BGL
			g = new BGLGraph();//if distribution changes, constructor can be passed the numvertices argument 
#endif
		vertextype = KDTREE;
		curAggrBlkId = 1;
		numProcs = num_processes(prg);		
#ifdef MESSAGE_AGGREGATION
		readyToFlush = false;
		numReadyToFlushUpdates = 0;
#endif
		//numVertices = 0;
#ifdef TRAVERSAL_PROFILE
		numPRootLeaves = 0;
#endif
#ifdef STATISTICS
		bufferStage = NULL;
		traverseCount = 0;
		pointsProcessed = 0;
#endif
#ifdef PERFCOUNTERS
		totTime = 0.0;
		compTime = 0.0;
#endif
		procRank = process_id(prg);
		nodeCountTable = new long int[numProcs];
		memset(nodeCountTable,0,sizeof(long int)*numProcs);
#ifdef LOAD_BALANCE
		//replicatedVertexCounter = 0;
		numUpdatesRequired = 0;
		readyToExit = false;
#endif
	}
	~GAL()
	{
		delete [] nodeCountTable;
		nodeCountTable = NULL;
	}

	int GAL_BuildSubTree(TIndices& pointRefs, GAL_Vertex* subtreeRoot,int height, int DOR, bool isLeft, const Box& boundingBox);
	bool MedianSplit(GAL_Vertex* intNode, int height, TIndices& pointRefs, TIndices& leftPoints, TIndices& rightPoints, Box& leftBox, Box& rightBox);
	bool SAHSplit(GAL_Vertex* intNode, int height, TIndices& pointRefs, TIndices& leftPoints, TIndices& rightPoints, Box& leftBox, Box& rightBox);
	void GAL_UpdateLocalNodeCount(int procId, int nodeCount);
	int GAL_GetNextProcessId(int subtreeHeight, long int pLeaf);
#ifdef LOAD_BALANCE
	GAL_Vertex* GAL_GetLocalVertex(long int localNode);
	int GAL_Aux_UpdatePLeaves(MsgUpdatePLeaves& msg, int pid);
	bool GAL_Aux_IsReadyToExit(){return (numUpdatesRequired == 0)?true:false;}
#endif
#ifdef SPAD_2
	void GALHelper_ReplicateSubtreeFromString(MsgReplicateSubtree& msgCloneSubtree, GAL_Vertex** pRoot, GAL_Vertex* pLeaf, int pLeafOwner);
	long GALHelper_GetRandom(int size);
	void GALHelper_SaveSubtreeAsString(GAL_Vertex* subtreeRoot, std::vector<std::string>& subtreeStr, long int parentId, int childNum);
#endif
	static GAL* instance;
	GAL_Graph* g;
	GAL_Vertex* rootNode; 
	int curAggrBlkId;
#ifdef MESSAGE_AGGREGATION
	std::map<long int, long int> aggrBuffer;
	//std::list<CompressedBlkObj> deFragTable;
	bool readyToFlush;
	int numReadyToFlushUpdates;
#endif
#ifdef PERFCOUNTERS
	uint64_t totTime;
	uint64_t compTime;
#endif


};



template<typename T>
void GAL_BroadCastObject(T& obj,GAL* g);
/*{
	for(int i=0;i<g->numProcs;i++)
	{
		if(i != g->procRank)
		{
			send_oob(g->pg,i,MESSAGE_DATA,obj); 
		}
	}
	return;

}*/
bool GAL_ReceiveObject(Photon& obj, GAL* g);
void GAL_BroadCastMessage(GAL*g, int msg);

#endif

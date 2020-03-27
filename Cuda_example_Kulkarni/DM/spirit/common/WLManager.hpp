/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef WLMANAGER_HPP
#define WLMANAGER_HPP
#include <boost/serialization/serialization.hpp>
#include "SPIRIT.hpp"
#define INVALID_BLOCK_ID -1
typedef std::vector<int> TIndices; //vector of index numbers (pointers) pointing to within the sub-block ( == blockStack entry)

/* Pseudo-Roots maintain a list of blocks that visit them. Only a subset of a block visited may proceed down the subtree rooted at a pseudo-root. Thus, the subset gets pushed onto block stack. 
 * However, when the subset completes its traversal, the entire block that visited earlier needs to be sent back to parent. Hence, the data structure 'SuperBlock' is defined. This structure contains
 * the entire block visited and a unique identifier- blockId. The same information can also be maintained on the block-stack. However, block stack management becomes cumbersome.
 **/ 
class SuperBlock
{
	public:
	TIndices block;
	TBlockId blockId;
	Vertex* nextNodeToVisit;
	char nextNodeProc;
	SuperBlock(TIndices& blk,TBlockId blkId, Vertex* ntv, int nnp):block(blk),blockId(blkId), nextNodeToVisit(ntv), nextNodeProc(nnp){}
};

typedef struct BSEntry
{
	TIndices block;
	Vertex* nodeToVisit;
	BSEntry* next;
	BSEntry():nodeToVisit(NULL),next(NULL){}
	~BSEntry()
	{
		block.erase(block.begin(),block.end());
	}
}BSEntry;

typedef struct BlockStack
{
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


typedef TPointVector::iterator TPointBlockIter; 
typedef BlockStack* BlockStackList;
typedef BlockStack* BlockStackListElem;
#ifdef MESSAGE_AGGREGATION
typedef std::vector<BlockStack*> TCBSet;
#endif

class WLManager
{
	int numPipelineElems;
	int numWorkItems;
	int bStackListSize;
	int offsetFactor;
	Optimizations& opts;
	TBlockId curBlockId;
	BlockStackList bStackList;
	BlockStackListElem curBlockStack;
	std::vector<std::pair<long int, long int> > subBlockPair;

	void Eff_AddBlockForTraversal(TIndices& pvRef, const Vertex* startNode, int curBlkId, Vertex* sib, int procRank);

	public:
	unsigned char* points;
	long int numPoints;
	long int end, start;
	long int totalNodesTraversed;
	
	WLManager(Optimizations& o, unsigned char* p, long int numPts, int of):opts(o),points(p),numPoints(numPts),offsetFactor(of){numWorkItems=0;totalNodesTraversed=0;}
	WLManager(Optimizations& o, unsigned char* p, long int numPoints, long int si, long int ei, int of);
	void ComputeNumberOfWorkBlocks();
	void InitializeState();
	int FillPipeline(Vertex* startNode, int procRank);
 	int GetNumberOfWorkItems();
	int RemoveBlockStack(SPIRITVisitor* vis);
	void SetAsCurrentBlockStack2(BlockStackListElem curBStack);
	BlockStackListElem GetCurrentBlockStack();
	bool VisitNode(Vertex* node, SPIRITVisitor* vis, Vertex* nextNodeToVisit, TIndices& leftBlock, TIndices& rightBlock, TBlockId& curBlkId);
	BlockStackListElem CreateBlockStack(const TIndices& blk, const Vertex* rNode, Vertex* sibNode, int numExpUpdates, BlockStackListElem& curBStack, int procRank);
	TBlockId GetContext(SPIRITVisitor* vis, TContextVector& lData);
	void FreeContext(TContextVector& lData);
	Vertex* PopFromCurrentBlockStackAndUpdate();
	void PushToBlockStackAndUpdate(SPIRITVisitor* vis, MsgTraverse& msg, int procRank);
	int SetContext(SPIRITVisitor* vis, const TContextVector& lData, TBlockId blkID, bool updateCurBlockStack=true);
	TBlockId DeleteFromSuperBlock(Vertex* pRoot, Vertex** nextNodeToVisit, int& nextNodeProc);
	bool IsLastFragment(Vertex* node, BlockStackListElem curBStack, TBlockId& parentBlkId);
	void SetAsCurrentBlockStack(const TBlockId& blkId);
	Vertex* GetNextNodeToVisit(Vertex* curNode, bool pop);
	void AddSubBlock(long int startIndex, long int endIndex);
	void PrepareWorkBlocks();
#ifdef MESSAGE_AGGREGATION
	BlockStackListElem CreateBufferStack(BlockStack* bufferStack, bool leftBuffer, int& curBufferSize, int procRank); 
	TBlockId GetBufferData(SPIRITVisitor* vis, TContextVector& lData, TContextVector& rData);
	int GetCompressedBlockIDs(TBlockId blockId, Context* lData, TCBSet& blkIdSet);
	bool IsBufferedBlock(long int bStack);
#endif
};


#endif

/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef CORRELATION_VISITOR_H
#define CORRELATION_VISITOR_H
#include "BGLKDTree.h"
#include"GAL.h"

class CorrelationVisitor : GALVisitor 
{
private:
	int procRank;
#ifndef OPT
	TPointVector& points;
#endif
	int blkIntakeLimit;
	int blockSize;
	float radius;
	long int start;
	long int end;
	int replicationFactor;
	int workLoad;
public:
	int numPipelineElems;

	TBlockId curBlockId;
	TIndices block; //subset of the block of points that correlate at any node.

	int bStackListSize;
	BlockStackList bStackList;
	BlockStackListIter curBlockStack;
	
	bool allBlocksVisited;
	int maxSize;
	int numWorkItems;
	CorrelationVisitor(int pid, TPointVector& pts, int blkSize, int blkIntake):procRank(pid), points(pts),numWorkItems(-1), numPipelineElems(0), blockSize(blkSize), blkIntakeLimit(blkIntake)
	{
		workLoad = 0;
		replicationFactor = 0;
		bStackListSize=0;
		bStackList = NULL;
		curBlockStack = NULL;
	}
	void SetRadius(float r){radius = r;}
	void Eff_AddBlockForTraversal(TIndices& pvRef, const GAL_Vertex* startNode, int curBlkId, GAL_Vertex* sib);
	~CorrelationVisitor(){}
	bool CanCorrelate(const GAL_Vertex* node, GAL_Vertex* nextNodeToVisit, TIndices& leftBlock, TIndices& rightBlock);
	void discover_vertex(const GAL_Vertex* node);
	void ComputeNumberOfWorkBlocks();
	void AssignWorkItemBounds(long int sI, long int eI, int rep){start=sI;end=eI;replicationFactor = rep;}
	int VisitDFS(GAL* g, int pid, std::vector<int> pipelineBufferSizes, int numBuffers);
	void UpdateClosest(Point& pt, const BGLTreeNode& node);
	//GALVisitor related methods
	bool GALVisitor_VisitNode(const GAL_Vertex* v, GAL_Vertex* nextNodeToVisit, TIndices& leftBlock, TIndices& rightBlock, TBlockId& curBlkId);
	int GALVisitor_GetNumberOfWorkItems();
	TBlockId GALVisitor_GetLocalData(TLocalDataVector& lData);
	int GALVisitor_SetLocalData(const TLocalDataVector& lData, TBlockId blkId, bool updateCurBlockStack = true);
	void GALVisitor_SetBlock(const TIndices& blk);
	void GALVisitor_PushToBlockStackAndUpdate(MsgTraverse& msg);
	void GALVisitor_UpdateBlockFromBlockStackTop(GAL_Vertex* searchNode=NULL);
	int GALVisitor_FillPipeline(GAL_Vertex* startNode);
	void GALVisitor_SetAsCurrentBlockStack(const TBlockId& blkId);
	int GALVisitor_RemoveBlockStack();
	GAL_Vertex* GALVisitor_PopFromCurrentBlockStackAndUpdate();
	TBlockId GALVisitor_DeleteFromSuperBlock(GAL_Vertex* pRoot, GAL_Vertex** nextNodeToVisit, int& nextNodeProc);
	//bool GALVisitor_IsSuperBlock(TBlockId blkId);
	BlockStackListIter GALVisitor_CreateBlockStack(const TIndices& blk, const GAL_Vertex* rNode, GAL_Vertex* sibNode, int numExpUpdates, BlockStackListIter& curBStack);
	void GALVisitor_UpdateCurrentBlockStackId(GAL_Vertex* pRoot);
	BlockStackListIter GALVisitor_GetCurrentBlockStack();
	void GALVisitor_SetAsCurrentBlockStack2(BlockStackListIter curBStack);
	bool GALVisitor_IsLastFragment(GAL_Vertex* node, BlockStackListIter curBStack, TBlockId& parentBlkId);	
#ifdef PERFCOUNTERS
	BlockStats GALVisitor_GetNumberOfBlocksInFlight();
	TBlockId GALVisitor_GetCurrentBlockId();
	long int GALVisitor_DetermineOrigBlkId(LocalData& ld);
#endif
	int GALVisitor_GetWorkerId(long int workItemId);
#ifdef MESSAGE_AGGREGATION
	BlockStackListIter GALVisitor_CreateBufferStack(BlockStack* bufferStack, bool leftBuffer, int& curBufferSize); 
	TBlockId GALVisitor_GetBufferData(TLocalDataVector& lData, TLocalDataVector& rData);
	int GALVisitor_GetCompressedBlockIDs(TBlockId blockId, LocalData& lData, TCBSet& blkIdSet);
	bool GALVisitor_IsBufferedBlock(long int blkStack);
#endif

};

#endif

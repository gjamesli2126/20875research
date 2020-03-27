#ifndef TRIANGLEKDTREE_VISITOR_H
#define TRIANGLEKDTREE_VISITOR_H
#include"GAL.h"
#include<shared/IntersectInfo.h>

/*class RayPlus{
Public:
Ray& ray;
IntersectInfo& info;
RayPlus(Ray& r, IntersectInfo& in):ray(r),in(info){}
};
typedef std::vector<RayPlus> TRayVector;*/
typedef std::vector<Ray> TRayVector;
typedef std::vector<IntersectInfo> TInfoVector;
typedef TRayVector::iterator TRayBlockIter;
typedef TInfoVector::iterator TInfoBlockIter;

class TriangleKDTreeVisitor : public GALVisitor 
{
private:
	int procRank;
	TRayVector& rays;
	TInfoVector& infos;
	GAL *gal;
	int blkIntakeLimit;
	int blockSize;

public:
	long int footPrint;
	int numPipelineElems;

	TBlockId curBlockId;
	TIndices block; //subset of the block of points that correlate at any node.

	int bStackListSize;
	BlockStackList bStackList;
	BlockStackListIter curBlockStack;
	
	int numWorkItems;
#ifdef LOAD_BALANCE
	long int totalWorkItems;
	int start;
	int end;
	int workLoad;
#endif
	TriangleKDTreeVisitor(int pid, GAL* g, TRayVector& rs, TInfoVector& ins):procRank(pid), rays(rs), gal(g), infos(ins), numWorkItems(-1), numPipelineElems(0)
	{
		footPrint = 0;
		blockSize = g->GetOpt_BlockSize();
		blkIntakeLimit = g->GetOpt_BlockIntakeLimit();
		numWorkItems = rays.size()/blockSize;
		if(rays.size() % blockSize)
			numWorkItems +=1;
		
#ifdef LOAD_BALANCE
		totalWorkItems = 0;
		start = 0;
		end=0;
		workLoad=0;
#endif
		bStackListSize=0;
		bStackList = NULL;
		curBlockStack = NULL;
	}
	void Eff_AddBlockForTraversal(TIndices& pvRef, const GAL_Vertex* startNode, int curBlkId, GAL_Vertex* sib);
	~TriangleKDTreeVisitor(){}
	bool CanCorrelate(GAL_Vertex* node, GAL_Vertex* nextNodeToVisit, TIndices& leftBlock, TIndices& rightBlock);
	void ComputeNumberOfWorkBlocks();

	//GALVisitor related methods
	int Traverse();
	bool GALVisitor_VisitNode(GAL_Vertex* v, GAL_Vertex* nextNodeToVisit, TIndices& leftBlock, TIndices& rightBlock, TBlockId& curBlkId);
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
	void GALVisitor_AddToSuperBlock(GAL_Vertex* pRoot, TLocalDataVector& lData, TBlockId tmpBlockId, GAL_Vertex* nextNodeToVisit, int nextNodeDesc);
	BlockStackListIter GALVisitor_CreateBlockStack(const TIndices& blk, const GAL_Vertex* rNode, GAL_Vertex* sibNode, int numExpUpdates, BlockStackListIter& curBStack);
	void GALVisitor_UpdateCurrentBlockStackId(GAL_Vertex* pRoot);
	BlockStackListIter GALVisitor_GetCurrentBlockStack();
	void GALVisitor_SetAsCurrentBlockStack2(BlockStackListIter curBStack);
	bool GALVisitor_IsLastFragment(GAL_Vertex* node, BlockStackListIter curBStack, TBlockId& parentBlkId);	
#ifdef PERFCOUNTERS
	BlockStats GALVisitor_GetNumberOfBlocksInFlight();
#endif
#ifdef MESSAGE_AGGREGATION
	BlockStackListIter GALVisitor_CreateBufferStack(BlockStack* bufferStack, bool leftBuffer, int& curBufferSize); 
	TBlockId GALVisitor_GetBufferData(TLocalDataVector& lData, TLocalDataVector& rData);
	int GALVisitor_GetCompressedBlockIDs(TBlockId blockId, LocalData& lData, TCBSet& blkIdSet);
	bool GALVisitor_IsBufferedBlock(long int blkStack);
#endif
#ifdef LOAD_BALANCE
	int GALVisitor_GetWorkerId(long int workItemId);
	TBlockId GALVisitor_GetCurrentBlockId();
	void GALVisitor_AssignWorkItems(long int startIndex, long int endIndex)
	{
		start=startIndex;
		end=endIndex;
		numWorkItems = (end-start)/blockSize;
		if((end-start) % blockSize)
			numWorkItems +=1;
	}
#endif

};

#endif

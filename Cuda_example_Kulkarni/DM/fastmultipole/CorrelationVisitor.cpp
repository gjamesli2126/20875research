/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include "CorrelationVisitor.h"
//#include <boost/graph/parallel/algorithm.hpp>
#ifdef STATISTICS2
int numBlocks;
double parOverhead=0.;
long int workDone;
#endif


void CorrelationVisitor::discover_vertex(const GAL_Vertex* node)

{
}

bool CorrelationVisitor::CanCorrelate(const GAL_Vertex* node, GAL_Vertex* nextNodeToVisit, TIndices& leftBlock, TIndices& rightBlock)
{
	return true;	
} 

void CorrelationVisitor::ComputeNumberOfWorkBlocks()
{
	numWorkItems = (end - start)/blockSize;
	if((end-start) % blockSize)
		numWorkItems +=1;
}

void CorrelationVisitor::Eff_AddBlockForTraversal(TIndices& pvRef, const GAL_Vertex* startNode, int curBlkId, GAL_Vertex* sib)
{
	BlockStack* tmpBlockStack;
	try
	{
		tmpBlockStack=new BlockStack();
	}
	catch(std::bad_alloc &ba)
	{
		printf("ERROR:MemoryFailure\n");
	}
	bStackListSize++;
#ifdef PERFCOUNTERS
	tmpBlockStack->entryNode=reinterpret_cast<long int>(startNode);
#endif
	tmpBlockStack->bStackId.second = reinterpret_cast<long int>(tmpBlockStack);
	tmpBlockStack->bStackId.first = procRank;
	tmpBlockStack->numFragments = 0;
	tmpBlockStack->parentBlockId.second = INVALID_BLOCK_ID;
	tmpBlockStack->parentBlockId.first = procRank;
	tmpBlockStack->parentBStack = NULL;
	
	BSEntry* stackElm;
	try
	{
		stackElm=new BSEntry();
	}
	catch(std::bad_alloc &ba)
	{
		printf("ERROR:MemoryFailure\n");
	}
	stackElm->block = pvRef;
	stackElm->nodeToVisit = sib;
	tmpBlockStack->bStack=stackElm;
	tmpBlockStack->next = bStackList;
	if(bStackList)
		bStackList->prev = tmpBlockStack;
	bStackList = tmpBlockStack;
	block.clear();
	block = pvRef;
	curBlockStack = tmpBlockStack;
	curBlockId = tmpBlockStack->bStackId;

}

int CorrelationVisitor::VisitDFS(GAL* g, int pid, std::vector<int> pipelineBufferSizes, int numBuffers)
{
	int ret;
	GAL_Vertex* startNode = g->GAL_GetRootNode();
	ret = g->GAL_Traverse(this, blockSize, numBuffers, pipelineBufferSizes);
	return ret;
}

/* Description: This function is called by the graph abstraction layer to update a point's state when the point visits a node.
 * Parameters: g - IN - reference to the graph.
 * 	 	node - IN - the node/vertex being visited.
 * 	 	leftBlock - OUT - the subset of points that decided to traverse the left subtree. Empty if there are no points.
 * 	 	rightBlock - OUT - the subset of points that decided to traverse the right subtree.Empty if there are no points.
 * Return Value: true if there is atleast one point that proceeds to visit either the left or right subtree.
 * Preconditions: 'block' contains the set of points that visit the 'node'.
 * Postconditions: the state of the points (number of nodes traversed, correlation value, closest_dist etc) in the current 'block' is updated.
 */

bool CorrelationVisitor::GALVisitor_VisitNode(const GAL_Vertex* node, GAL_Vertex* nextNodeToVisit, TIndices& leftBlock, TIndices& rightBlock, TBlockId& curBlkId)
{
	bool ret = false;
	//std::cout<<process_id((*g).process_group())<<": pot being checked: ("<<(pot).pt[0]<<", "<<(pot).pt[1]<<")"<<std::endl;
	discover_vertex(node);
	ret = CanCorrelate(node,nextNodeToVisit,leftBlock, rightBlock);
	curBlkId = curBlockId;
	return ret;
		
}


int CorrelationVisitor::GALVisitor_FillPipeline(GAL_Vertex* startNode)
{
	if(numWorkItems <= 0)
		return STATUS_NO_WORKITEMS;

	if(numPipelineElems == blkIntakeLimit)
		return STATUS_PIPELINE_FULL;

	numPipelineElems++;


	/*TIndices tmpBlock;
	long int endIndx = (numWorkItems)*blockSize;
	if(endIndx > points.size())
		endIndx = points.size();
	long int startIndx = (numWorkItems-1) * blockSize;
	for(long int i=startIndx;i<endIndx;i++)
		tmpBlock.push_back(i);
	Eff_AddBlockForTraversal(tmpBlock,startNode, numWorkItems, NULL);*/

	long int startIndx = (numWorkItems-1) * blockSize;
	long int endIndx = (numWorkItems) * blockSize;
	startIndx = start+startIndx;
	endIndx = start + endIndx;
	if(endIndx > end)
		endIndx = end;
	TIndices tmpBlock(endIndx - startIndx);

	for(long int i=0;i<(endIndx-startIndx);i++)
	{
		tmpBlock[i]=startIndx+i;
	}
	Eff_AddBlockForTraversal(tmpBlock,startNode, start+numWorkItems, NULL);

	numWorkItems--;

	return STATUS_SUCCESS;
	
}

int CorrelationVisitor::GALVisitor_GetNumberOfWorkItems()
{
	return bStackListSize;
}


int CorrelationVisitor::GALVisitor_SetLocalData(const TLocalDataVector& lData, TBlockId origBlkId, bool updateCurBlockStack)
{
	TIndices tmpIndices;

	block.clear();
	
	curBlockId = origBlkId;
			Point* iter = points;//.begin();
	for(int i=0;i<lData.size();i++)
	{
		LocalData l;
		(iter +lData[i].index)->nodesTraversed = lData[i].nodesTraversed;
	}
		if(updateCurBlockStack)
	{
			block.clear();
			curBlockId = origBlkId;
			curBlockStack = reinterpret_cast<BlockStack*>(origBlkId.second);
#if 0
			BlockStackListIter tmpIter = bStackList;
			while(tmpIter != NULL)
			{
				long int tmpPRoot = reinterpret_cast<long int>((tmpIter->blkId).first);
				if(((tmpIter->blkId).second == origBlkId.second) && (tmpPRoot == origBlkId.first))
					break;
				tmpIter = tmpIter->next;
			}
			/*if(tmpIter == NULL)
			{
				FILE* fp = fopen("log.txt","w+");
				BlockStackListIter tmpIter = bStackList;
				while(tmpIter != NULL)
				{
					long int tmpPRoot = reinterpret_cast<long int>((tmpIter->blkId).first);
					fprintf(fp,"ERROR:pRoot: %ld, BlkId: %ld \n", tmpPRoot,(tmpIter->blkId).second);
					tmpIter = tmpIter->next;
				}
				fprintf(fp,"ERROR:pRoot: %ld, BlkId: %ld \n", origBlkId.first, origBlkId.second);
				fclose(fp);
				GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(origBlkId.first);
				return -1;
			}*/
			assert(tmpIter != NULL);
			curBlockStack = tmpIter;
#endif
			block = curBlockStack->bStack->block;
	}

#ifdef STATISTICS2
	if(updateCurBlockStack)
	{
	double endTime=clock();
	double elapsedTime = endTime-curBlockStack->startTime;
	curBlockStack->startTime =clock();
	curBlockStack->parOverhead += elapsedTime;
	}
#endif
	return 0;
}

/* Description: This function should be called only when leaving a process boundary. i.e at pseudoLeaves.
 * This function removes the blockID from traversalready queue and returns the locally computed values of 
 * all the elements that are pointed to by the current block indices.
 * Parameters: lData - reference to a vector that contains locally computed data
 * Return Value: blockId of the current block.
 */
TBlockId CorrelationVisitor::GALVisitor_GetLocalData(TLocalDataVector& lData)
{
	bool flag = false;

		Point* iter = points;//.begin();

	//block contains indices relative to the beginning of the block.
	for(int i=0;i<block.size();i++)
	{
		LocalData l;
		l.nodesTraversed = (iter +block[i])->nodesTraversed;
			l.index = block[i];
			lData.push_back(l);
		}

#ifdef STATISTICS2
	curBlockStack->startTime = clock();
#endif
	return curBlockId;
}


void CorrelationVisitor::GALVisitor_UpdateBlockFromBlockStackTop(GAL_Vertex* searchNode)
{

	if(curBlockStack->bStack)
	{
		block.clear();
		block = curBlockStack->bStack->block;
	}	
}

bool CorrelationVisitor::GALVisitor_IsLastFragment(GAL_Vertex* node, BlockStackListIter curBStack, TBlockId& parentBlkId)
{
	bool ret = false;
	if(curBStack->bStack == NULL)
		printf("Debug Break\n");
	BlockStack* tmpBStack = reinterpret_cast<BlockStack *>(curBStack->parentBStack);
#ifdef MESSAGE_AGGREGATION
	if(node->pseudoRoot)
	{
		//split point.
		BSEntry* bsEntryRoot = curBStack->bStack->next;
		if(bsEntryRoot && (bsEntryRoot->next == NULL) && tmpBStack)
		{
			parentBlkId = (curBStack->parentBlockId);
			if(tmpBStack->numFragments == 1)
				ret = true;
		}
	}
#endif	
	if((curBStack->bStack->next == NULL ) && tmpBStack)
	{
		parentBlkId = (curBStack->parentBlockId);
		if(tmpBStack->numFragments == 1)
			ret = true;
	}
	return ret;
}	


BlockStackListIter CorrelationVisitor::GALVisitor_GetCurrentBlockStack()
{
	return curBlockStack;
}
void CorrelationVisitor::GALVisitor_SetAsCurrentBlockStack2(BlockStackListIter curBStack)
{
	curBlockStack = curBStack;
	GALVisitor_UpdateBlockFromBlockStackTop();
	curBlockId = curBlockStack->bStackId;
}

void CorrelationVisitor::GALVisitor_SetAsCurrentBlockStack(const TBlockId& blockStackId)
{
	curBlockStack = reinterpret_cast<BlockStack*>(blockStackId.second);
	curBlockId = curBlockStack->bStackId;
	GALVisitor_UpdateBlockFromBlockStackTop();
}

int CorrelationVisitor::GALVisitor_RemoveBlockStack()
{
	int ret = 0;
		BlockStack* parentBStack = reinterpret_cast<BlockStack *>(curBlockStack->parentBStack);
		if(parentBStack)
		{
			if(parentBStack->numFragments > 0)
				parentBStack->numFragments--;
			else
			{
				printf("debug break\n");
				return 0;
			}
		}
		
		/*if((curBlockStack->blkId).second == 58721335)
		{
			printf("Required BlockStackBeing Removed\n");
			ret = -1;
		}*/	

		if(curBlockStack->parentBStack == NULL)
		{
#ifdef STATISTICS2
			if(curBlockStack->parentBlockId.second == INVALID_BLOCK_ID)
			{
				//parOverhead +=curBlockStack->parOverhead;
				if(parOverhead < curBlockStack->parOverhead)
					parOverhead =curBlockStack->parOverhead;
				//code for printing individual block waiting times.
				/*char filename[16];
				FILE* fp;
				sprintf(filename,"log%d.txt",procRank);	
				fp=fopen(filename,"a+");
				fprintf(fp,"%f ",curBlockStack->parOverhead/CLOCKS_PER_SEC);
				fclose(fp);*/

				numBlocks++;
			}
#endif
			if(curBlockStack->bStack && curBlockStack->bStack->next)
				printf("ERROR: Wrong block being cleared\n");
			numPipelineElems--;
			if(numPipelineElems < 0)
				numPipelineElems = 0;
		}
		else
		{
			/*if(curBlockStack->bStack)
				printf("ERROR: current block being cleared\n");*/
		}
		BlockStackListIter prevIter=bStackList,tmpIter = bStackList;
		if(curBlockStack == bStackList)
		{
			bStackList = bStackList->next;
			if(bStackList)
				bStackList->prev = NULL;
			/*if(procRank == 3)
			{
			//FILE* fp = fopen("log3.txt","w+");
			//printf("d: pRoot: %ld, BlkId: %ld \n", reinterpret_cast<long int>(curBlockStack->blkId.first),curBlockStack->blkId.second);
			//fclose(fp);
			}*/
			delete curBlockStack;
		}
		else
		{
			/*while(tmpIter != NULL)
			{
				if(tmpIter == curBlockStack)
				{
					prevIter->next=tmpIter->next;
					delete tmpIter;
					break;
				}
				prevIter = tmpIter;
				tmpIter = tmpIter->next;
			}*/
			
			prevIter = curBlockStack->prev;
			tmpIter = curBlockStack->next;
			if(prevIter)
				prevIter->next = tmpIter;
			if(tmpIter)
				tmpIter->prev = prevIter;
			delete curBlockStack;
		}
		bStackListSize--;
		return ret;
}


GAL_Vertex* CorrelationVisitor::GALVisitor_PopFromCurrentBlockStackAndUpdate()
{
	GAL_Vertex* ret = NULL;
	{
		if(curBlockStack->bStack)
		{
			BSEntry* tmp=curBlockStack->bStack;	
			ret = tmp->nodeToVisit;
			curBlockStack->bStack = tmp->next;
			delete tmp;
		}
		/*else
		{
			assert(0);
		}*/
		if(curBlockStack->bStack)
			block = curBlockStack->bStack->block;
	}

	return ret;
}

void CorrelationVisitor::GALVisitor_PushToBlockStackAndUpdate(MsgTraverse& msg)
{
	block.clear();

	//update the nodesTraversed value for the point Block.
	//obtain the handle to the block. blkStart is the id of the block.
	//if(blkStart >= 0)
	{
		Point* iter = points;//.begin();

		for(int i=0;i<msg.l.size();i++)
		{
			LocalData l;
			block.push_back(msg.l[i].index);
			(iter +msg.l[i].index)->nodesTraversed = msg.l[i].nodesTraversed;
		}
	}
	

	BlockStack* tmpBlockStack;
	BSEntry* bsEntry;
	try{
		tmpBlockStack = new BlockStack();
		bsEntry = new BSEntry;
	}
	catch(std::bad_alloc &ba)
	{
		printf("ERROR:MemoryFailure\n");
	}
	bsEntry->block = block;
	bsEntry->nodeToVisit = reinterpret_cast<GAL_Vertex*>(msg.pSibling);
#ifdef PERFCOUNTERS
	tmpBlockStack->entryNode=msg.pRoot;
#endif
	tmpBlockStack->bStack=bsEntry;
	bStackListSize++;
	tmpBlockStack->bStackId.second = reinterpret_cast<long int>(tmpBlockStack);
	tmpBlockStack->bStackId.first = procRank;
	tmpBlockStack->numFragments = 0;
	tmpBlockStack->parentBlockId=msg.blkStart;
	tmpBlockStack->parentBStack = NULL;
	tmpBlockStack->nextNodePid = msg.siblingDesc;
	curBlockId =  tmpBlockStack->bStackId;
	//pair this iterator with pv and then add it to blockstack-point vector map.
	/*bStackList.push_front(tmpBlockStack);
	curBlockStack = bStackList.begin();*/
	tmpBlockStack->next = bStackList;
	if(bStackList)
		bStackList->prev = tmpBlockStack;
	bStackList = tmpBlockStack;
	curBlockStack = bStackList;
}


/* Description: This function is called by the Graph Abstraction Layer when the traversal begins/leaves any node.
 * Before the traversal begins, the set of points that are visiting/traversing the nodes needs to be defined.
 * Returns: None
 * Precondition: Since only the indices/references are passed, it is assumed that the actual points are stored in memory.
 * Parameters: blk - set of integers that denote the index values of points in the block.
 * Post condition: 'block' is set.
 */

void CorrelationVisitor::GALVisitor_SetBlock(const TIndices& blk)
{
	block.clear();
	TIndices::const_iterator iter = blk.begin();
	while(iter != blk.end())
	{
		block.push_back(*iter);
		iter++;
	}
}

/* Description: This function is called by the Graph Abstraction Layer when the traversal begins at a pseudo root.
 * Returns: None
 * Parameters: pRoot - reference to the pseudo-root
 * Post condition: adds the tuple <block, block ID> to the list maintained by pseudoRoot.
 */
void CorrelationVisitor::GALVisitor_AddToSuperBlock(GAL_Vertex* pRoot, TLocalDataVector& lData, TBlockId tmpBlockId, GAL_Vertex* nextNodeToVisit, int nextNodeProc)
{
}

/* Description: This function is called by the Graph Abstraction Layer when the traversal leaves a pseudo-root and resumes at either the parent or sibling.
 * Returns: None
 * Precondition: 'block' and 'block ID' must correspond to the current traversal.
 * Parameters: pRoot - reference to the pseudo-root
 * Post condition: 'block' is set.
 * 		   deletes the tuple <block, block ID> to the list maintained by pseudoRoot.
 */
TBlockId CorrelationVisitor::GALVisitor_DeleteFromSuperBlock(GAL_Vertex* pRoot, GAL_Vertex** nodeToVisit, int& nextNodeProc)
{
	if(curBlockStack->bStack)
	{
		*nodeToVisit = curBlockStack->bStack->nodeToVisit;
		block = curBlockStack->bStack->block;
	}
	nextNodeProc = curBlockStack->nextNodePid;
	return curBlockStack->parentBlockId;
}

BlockStackListIter CorrelationVisitor::GALVisitor_CreateBlockStack(const TIndices& blk, const GAL_Vertex* rNode, GAL_Vertex* sibNode, int numExpUpdates, BlockStackListIter& curBStack)
{
	curBStack->numFragments++;
	block.clear();
	block = blk;
	BlockStack* tmpBlockStack;
	BSEntry* bsEntry;
	try
	{
		tmpBlockStack = new BlockStack();
		bsEntry = new BSEntry;
	}
	catch(std::bad_alloc &ba)
	{
		printf("ERROR:MemoryFailure\n");
	}
	bStackListSize++;
	bsEntry->block = block;
	bsEntry->nodeToVisit = sibNode;
#ifdef PERFCOUNTERS
	tmpBlockStack->entryNode=reinterpret_cast<long int>(rNode);
#endif
	tmpBlockStack->bStackId.second = reinterpret_cast<long int>(tmpBlockStack);
	tmpBlockStack->bStackId.first = procRank;
	tmpBlockStack->parentBlockId = curBStack->bStackId;
	tmpBlockStack->parentBStack = (void *)(&(*curBStack));
	curBlockId = tmpBlockStack->bStackId;
	tmpBlockStack->bStack=bsEntry;
	tmpBlockStack->next = bStackList;
	if(bStackList)
		bStackList->prev = tmpBlockStack;
	bStackList = tmpBlockStack;
	curBlockStack = tmpBlockStack;
	/*bStackList.push_front(tmpBlockStack);
	curBlockStack = bStackList.begin();*/
	return curBlockStack;
}

void CorrelationVisitor::GALVisitor_UpdateCurrentBlockStackId(GAL_Vertex* pRoot)
{
}

/*#ifdef PERFCOUNTERS
BlockStats CorrelationVisitor::GALVisitor_GetNumberOfBlocksInFlight()
{
	BlockStats ret;
	int uniqBStacks = 0;
	std::set<long int> uniqBStackSet;
	BlockStackListIter bspIter;
	for(bspIter = bStackList;bspIter!=NULL;bspIter=bspIter->next)
	{
		if(EXTRACT_SIGNBIT((bspIter->blkId).second))
			continue;
		if(uniqBStackSet.find((bspIter->blkId).second) == uniqBStackSet.end())
		{
			uniqBStackSet.insert((bspIter->blkId).second);
			uniqBStacks++;
		}
	}
	ret.numUniqBlocks = uniqBStacks;
	return ret;
}
#endif*/

#ifdef MESSAGE_AGGREGATION
BlockStackListIter CorrelationVisitor::GALVisitor_CreateBufferStack(BlockStack* bufferStack, bool leftBuffer, int& curBufferSize)
{
	BlockStack* tmpBlockStack;
	BSEntry* bsEntry;
	if(bufferStack == NULL)
	{
		try
		{
			tmpBlockStack = new BlockStack();
			bsEntry = new BSEntry;
		}
		catch(std::bad_alloc &ba)
		{
			printf("ERROR:MemoryFailure\n");
		}
		bStackListSize++;
		bsEntry->block = block;
		tmpBlockStack->bStackId.second = reinterpret_cast<long int>(tmpBlockStack);
		tmpBlockStack->bStackId.first = procRank;
		if(leftBuffer)
		{
			tmpBlockStack->leftBuffer.push_back(curBlockStack);
			tmpBlockStack->bStack=bsEntry;
			tmpBlockStack->bStack->next = new BSEntry;
		}
		else
		{
			tmpBlockStack->bStack=new BSEntry;
			tmpBlockStack->bStack->next=bsEntry;
			tmpBlockStack->rightBuffer.push_back(curBlockStack);
		}
		tmpBlockStack->next = bStackList;
		if(bStackList)
			bStackList->prev = tmpBlockStack;
		bStackList = tmpBlockStack;
		tmpBlockStack->isBuffer=true;
	}
	else
	{
		tmpBlockStack = bufferStack;
		if(leftBuffer)
		{
			tmpBlockStack->leftBuffer.push_back(curBlockStack);
			(bufferStack->bStack)->block.insert((bufferStack->bStack)->block.begin(), block.begin(), block.end());
		}
		else
		{
			(bufferStack->bStack->next)->block.insert((bufferStack->bStack->next)->block.begin(), block.begin(), block.end());
			tmpBlockStack->rightBuffer.push_back(curBlockStack);
		}
	}
	curBufferSize = tmpBlockStack->bStack->block.size() + tmpBlockStack->bStack->next->block.size();
	return tmpBlockStack;
} 


TBlockId CorrelationVisitor::GALVisitor_GetBufferData(TLocalDataVector& lData, TLocalDataVector& rData)
{
	bool flag = false;

	Point* iter = points;//.begin();
	//block contains indices relative to the beginning of the block.
	for(int i=0;i<block.size();i++)
	{
		LocalData l;
		l.nodesTraversed = (iter +block[i])->nodesTraversed;
		l.index = block[i];
		lData.push_back(l);
	}

	block.clear();
	iter=points;//.begin();
	block = curBlockStack->bStack->next->block;
	for(int i=0;i<block.size();i++)
	{
		LocalData l;
		l.nodesTraversed = (iter +block[i])->nodesTraversed;
		l.index = block[i];
		rData.push_back(l);
	}
	
	if(lData.size() > 0)
		curBlockStack->numFragments++;
	if(rData.size() > 0)
		curBlockStack->numFragments++;
	return curBlockId;	

}

int CorrelationVisitor::GALVisitor_GetCompressedBlockIDs(TBlockId blockId, LocalData& lData, TCBSet& blkIdSet)
{
	BlockStack* bufferStack = reinterpret_cast<BlockStack*>(blockId.second);
	bool found = false;

	if(bufferStack->isBuffer)
	{
		bufferStack->numFragments--;
		BSEntry* tmp = bufferStack->bStack;
		for(int i=0;i<tmp->block.size();i++)
		{
			if(lData.index == tmp->block[i])
			{
				found = true;
				break;
			}
		}
		if(!found)
		{
			blkIdSet=bufferStack->rightBuffer;
			bufferStack->rightBuffer.clear();
		}
		else
		{
			blkIdSet=bufferStack->leftBuffer;
			bufferStack->leftBuffer.clear();
		}	
		
		if(bufferStack->numFragments == 0)
		{
			delete bufferStack->bStack->next;
			delete bufferStack->bStack;
			bufferStack->bStack=NULL;
			BlockStack* prevIter = bufferStack->prev;
			BlockStack* tmpIter = bufferStack->next;
			if(prevIter)
				prevIter->next = tmpIter;
			if(tmpIter)
				tmpIter->prev = prevIter;
			if(bStackList == bufferStack)
				bStackList = bufferStack->next;
			delete bufferStack;
			bStackListSize--;
		}

#ifdef STATISTICS2
		TCBSet::iterator sIter = blkIdSet.begin();
		for(;sIter!=blkIdSet.end();sIter++)
		{
			double endTime=clock();
			double elapsedTime = endTime-(*sIter)->startTime;
			(*sIter)->startTime =clock();
			(*sIter)->parOverhead += elapsedTime;
		}
#endif
		return STATUS_SUCCESS;
	}

	return STATUS_FAILURE;
}

bool CorrelationVisitor::GALVisitor_IsBufferedBlock(long int blkStack)
{
	BlockStack* bStack = reinterpret_cast<BlockStack*>(blkStack);
	return bStack->isBuffer;
}
#endif

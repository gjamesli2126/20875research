/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include "WLManager.hpp"
#include "Optimizations.hpp"

WLManager::WLManager(Optimizations& o, unsigned char* p, long int numPts, long int si, long int ei, int of):opts(o)
{
	points = p;
	numPoints = numPts;
	numPipelineElems=0;
	bStackListSize = 0;
	bStackList = NULL;
	start = si;
	end = ei;
	totalNodesTraversed=0;
	offsetFactor = of;
	ComputeNumberOfWorkBlocks();
}

void WLManager::InitializeState()
{
	numPipelineElems=0;
	bStackListSize = 0;
	bStackList = NULL;
	ComputeNumberOfWorkBlocks();	
}
void WLManager::ComputeNumberOfWorkBlocks()
{
	numWorkItems = (end - start)/opts.blockSize;
	if((end-start) % opts.blockSize)
		numWorkItems +=1;
}

void WLManager::AddSubBlock(long int startIndex, long int endIndex)
{
	numWorkItems += (endIndex - startIndex);
	subBlockPair.push_back(std::make_pair(startIndex, endIndex));
	//printf("Adding block %d %d\n",startIndex, endIndex);
}

void WLManager::PrepareWorkBlocks()
{
	numWorkItems = ceil(numWorkItems / (float)(opts.blockSize));
	numPipelineElems=0;
	bStackListSize = 0;
	bStackList = NULL;
}

void WLManager::Eff_AddBlockForTraversal(TIndices& pvRef, const Vertex* startNode, int curBlkId, Vertex* sib, int procRank)
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
	curBlockStack = tmpBlockStack;
	curBlockId = tmpBlockStack->bStackId;

}

/* Description: This function is called by the graph abstraction layer to update a point's state when the point visits a node.
 * Parameters: g - IN - reference to the graph.
 * 	 	node - IN - the node/vertex being visited.
 * 	 	leftBlock - OUT - the subset of points that decided to traverse the left subtree. Empty if there are no points.
 * 	 	rightBlock - OUT - the subset of points that decided to traverse the right subtree.Empty if there are no points.
 * Return Value: true if there is atleast one point that proceeds to visit either the left or right subtree.
 * Preconditions: 'block' contains the set of points that visit the 'node'.
 * Postconditions: the state of the points (number of nodes traversed, correlation value, tau etc) in the current 'block' is updated.
 */

bool WLManager::VisitNode(Vertex* node, SPIRITVisitor* vis, Vertex* nextNodeToVisit, TIndices& leftBlock, TIndices& rightBlock, TBlockId& curBlkId)
{
	bool takeLeftBranch=false, takeRightBranch=false;
	status ret;
#ifdef TRAVERSAL_PROFILE
	int bSize = curBlockStack->bStack->block.size();
	(const_cast<Vertex*>(node))->pointsVisited += bSize;
#endif
	TIndices::iterator iter = curBlockStack->bStack->block.begin();
	for(;iter!=curBlockStack->bStack->block.end();iter++)
	{
		Point* p = (Point*)(points+(*iter * offsetFactor));
		/*if(p->id == 6)
			printf("break\n");*/
		ret = vis->EvaluateVertex(node, p);
		if((ret == TRUNC) || (node->leaf))
		{
			continue;
		}
		else if(ret == LEFT)
		{
			leftBlock.push_back(*iter);
			takeLeftBranch = true;
		}
		else
		{
			rightBlock.push_back(*iter);
			takeRightBranch = true;
		}
	}

	if(!takeRightBranch && !takeLeftBranch)
		return false;
	else
	{
		BSEntry* stackElm;
		try
		{
			stackElm = new BSEntry();
		}
		catch(std::bad_alloc &ba)
		{
			printf("ERROR:MemoryFailure\n");
		}
		stackElm->nodeToVisit = nextNodeToVisit;
		if(!takeLeftBranch || !takeRightBranch)
		{
			if(takeLeftBranch)
			{
				stackElm->block = leftBlock;
			}
			else if(takeRightBranch)
			{
				stackElm->block = rightBlock;
			}	
		}
		stackElm->next = curBlockStack->bStack;
		curBlockStack->bStack=stackElm;
	}
	
	curBlkId = curBlockId;
	return true;
		
}


int WLManager::FillPipeline(Vertex* startNode, int procRank)
{
	if(numWorkItems <= 0)
		return STATUS_NO_WORKITEMS;

	if(numPipelineElems == opts.blkIntakeLimit)
		return STATUS_PIPELINE_FULL;

	numPipelineElems++;

	if(opts.distribution == BLOCK_DIST)
	{
		//block distribution
		long int startIndx = (numWorkItems-1) * opts.blockSize;
		long int endIndx = (numWorkItems) * opts.blockSize;
		startIndx = start+startIndx;
		endIndx = start + endIndx;
		if(endIndx > end)
			endIndx = end;
		//printf("startIndx %ld  endIndx %ld\n",startIndx, endIndx);
		TIndices tmpBlock(endIndx - startIndx);

		for(long int i=0;i<(endIndx-startIndx);i++)
		{
			tmpBlock[i]=startIndx+i;
		}
		Eff_AddBlockForTraversal(tmpBlock,startNode, start+numWorkItems, NULL, procRank);
	}
	else if(opts.distribution == BLOCK_CYCLIC)
	{
		int j=0;
		//block-cyclic 
		int numElemsDispatched = 0;
		TIndices tmpBlock(opts.blockSize);
		while((numElemsDispatched < opts.blockSize) && (subBlockPair.size()>0))
		{
			long int i=0;
			long int startIndex = subBlockPair[0].first;
			long int endIndex = subBlockPair[0].second;
			for(i=startIndex;i<endIndex;i++,j++)
			{
				tmpBlock[j]=i;
				numElemsDispatched++;
				if(numElemsDispatched == opts.blockSize)
				{
					if(i!=endIndex-1)
						subBlockPair[0].first = i+1;
					break;
				}
			}
			if(i>=endIndex-1)
				subBlockPair.erase(subBlockPair.begin());
		}
		tmpBlock.erase(tmpBlock.begin()+numElemsDispatched, tmpBlock.end());
		Eff_AddBlockForTraversal(tmpBlock,startNode, numWorkItems, NULL, procRank);
	}

	numWorkItems--;

	return STATUS_SUCCESS;
	
}

int WLManager::GetNumberOfWorkItems()
{
	return bStackListSize;
}


int WLManager::SetContext(SPIRITVisitor* vis, const TContextVector& lData, TBlockId origBlkId, bool updateCurBlockStack)
{
	TIndices tmpIndices;

	curBlockId = origBlkId;
	//Aggregate block. Decompress.
	for(int i=0;i<lData.size();i++)
	{
		Point* p = (Point*)(points+(lData[i]->index * offsetFactor));
		vis->SetContext(p, lData[i]);
	}

	if(updateCurBlockStack)
	{
		curBlockId = origBlkId;
		curBlockStack = reinterpret_cast<BlockStack*>(origBlkId.second);
	}

	return 0;
}

/* Description: This function should be called only when leaving a process boundary. i.e at pseudoLeaves.
 * returns the locally computed values of 
 * all the elements that are pointed to by the current block indices.
 * Parameters: lData - reference to a vector that contains locally computed data
 * Return Value: blockId of the current block.
 */
TBlockId WLManager::GetContext(SPIRITVisitor* vis, TContextVector& lData)
{
	bool flag = false;

	TIndices::iterator iter = curBlockStack->bStack->block.begin();
	for(;iter!=curBlockStack->bStack->block.end();iter++)
	{
		Point* p = (Point*)(points+(*iter * offsetFactor));
		Context *l=vis->GetContext(p);
		l->index = *iter;
		lData.push_back(l);
	}
	return curBlockId;	
}

void WLManager::FreeContext(TContextVector& lData)
{
	TContextVector::iterator iter = lData.begin();
	while(iter!=lData.end())
	{
		delete *iter;
		iter++;
	}
}

bool WLManager::IsLastFragment(Vertex* node, BlockStackListElem curBStack, TBlockId& parentBlkId)
{
	bool ret = false;
	assert(curBStack->bStack);
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


BlockStackListElem WLManager::GetCurrentBlockStack()
{
	return curBlockStack;
}
void WLManager::SetAsCurrentBlockStack2(BlockStackListElem curBStack)
{
	curBlockStack = curBStack;
	curBlockId = curBlockStack->bStackId;
}

void WLManager::SetAsCurrentBlockStack(const TBlockId& blockStackId)
{
	curBlockStack = reinterpret_cast<BlockStack*>(blockStackId.second);
	curBlockId = curBlockStack->bStackId;
}

int WLManager::RemoveBlockStack(SPIRITVisitor* vis)
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
	
	if(curBlockStack->parentBStack == NULL)
	{
		if(curBlockStack->bStack && curBlockStack->bStack->next)
			printf("ERROR: Wrong block being cleared\n");
		if(curBlockStack->bStack)
		{
			BSEntry* bStack = curBlockStack->bStack;
			if(curBlockStack->parentBlockId.second == INVALID_BLOCK_ID)
			{
				numPipelineElems--;
				if(numPipelineElems < 0)
					numPipelineElems = 0;
				//printf("Top block cleared\n");
				TIndices::iterator iter = bStack->block.begin();
				for(;iter!=bStack->block.end();iter++)
				{
					Point* p = (Point*)(points+(*iter * offsetFactor));
					totalNodesTraversed += p->nodesTraversed;
					vis->ProcessResult(p);
					//printf("% ld %ld \n",p->id,p->nodesTraversed);
				}
				//printf("\n");
			}
		}
	}
	if(curBlockStack->bStack)
	{
		//printf("ERROR: current block being cleared\n");
		delete curBlockStack->bStack;
		curBlockStack->bStack=NULL;
	}

	BlockStackListElem prevIter=bStackList,tmpIter = bStackList;
	if(curBlockStack == bStackList)
	{
		bStackList = bStackList->next;
		if(bStackList)
			bStackList->prev = NULL;
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


Vertex* WLManager::PopFromCurrentBlockStackAndUpdate()
{
	Vertex* ret = NULL;
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

	return ret;
}

void WLManager::PushToBlockStackAndUpdate(SPIRITVisitor* vis, MsgTraverse& msg, int procRank)
{
	//update the nodesTraversed value for the point Block.
	//obtain the handle to the block. blkStart is the id of the block.

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
	for(int i=0;i<msg.l.size();i++)
	{
		bsEntry->block.push_back(msg.l[i]->index);
		Point* p = (Point*)(points+(msg.l[i]->index * offsetFactor));
		vis->SetContext(p,msg.l[i]);
	}
	
	bsEntry->nodeToVisit = reinterpret_cast<Vertex*>(msg.pSibling);
	tmpBlockStack->bStack=bsEntry;
	bStackListSize++;
	tmpBlockStack->bStackId.second = reinterpret_cast<long int>(tmpBlockStack);
	tmpBlockStack->bStackId.first = procRank;
	tmpBlockStack->numFragments = 0;
	tmpBlockStack->parentBlockId=msg.blkStart;
	tmpBlockStack->parentBStack = NULL;
	tmpBlockStack->nextNodePid = msg.siblingDesc;
	curBlockId =  tmpBlockStack->bStackId;
	tmpBlockStack->next = bStackList;
	if(bStackList)
		bStackList->prev = tmpBlockStack;
	bStackList = tmpBlockStack;
	curBlockStack = bStackList;

}


/* Description: This function is called by the Graph Abstraction Layer when the traversal leaves a pseudo-root and resumes at either the parent or sibling.
 * Returns: None
 * Precondition: 'block' and 'block ID' must correspond to the current traversal.
 * Parameters: pRoot - reference to the pseudo-root
 * Post condition: 'block' is set.
 * 		   deletes the tuple <block, block ID> to the list maintained by pseudoRoot.
 */
TBlockId WLManager::DeleteFromSuperBlock(Vertex* pRoot, Vertex** nodeToVisit, int& nextNodeProc)
{
	if(curBlockStack->bStack)
	{
		*nodeToVisit = curBlockStack->bStack->nodeToVisit;
	}
	nextNodeProc = curBlockStack->nextNodePid;
	return curBlockStack->parentBlockId;
}

BlockStackListElem WLManager::CreateBlockStack(const TIndices& blk, const Vertex* rNode, Vertex* sibNode, int numExpUpdates, BlockStackListElem& curBStack, int procRank)
{
	curBStack->numFragments++;
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
	bsEntry->block = blk;
	bsEntry->nodeToVisit = sibNode;

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
	return curBlockStack;
}

Vertex* WLManager::GetNextNodeToVisit(Vertex* curNode, bool pop)
{
	Vertex* ret = NULL;
	if(curNode->parent)
	{
		Vertex* parent = curNode->parent;
		for(int j=0;j<8;j++)
		{
			if(parent->pChild[j] == curNode)
			{
				for(int i=j+1;i<8;i++)
				{
					if(parent->pChild[i] != NULL)
					{
						ret = parent->pChild[i];
						break;
					}
				}
				break;
			}
		}
	}
	if(pop)
	{
		if((ret == NULL) && curBlockStack->bStack)
		{
			//Remove the stack entry only if this is not the last entry.
			//The last entry is removed RemoveBlockStack after visitor has processed the result.
			BSEntry* tmp=curBlockStack->bStack;	
			if(tmp->next)
			{
				curBlockStack->bStack = tmp->next;
				delete tmp;
			}
		}
	}
	return ret;
}

#ifdef MESSAGE_AGGREGATION
BlockStackListElem WLManager::CreateBufferStack(BlockStack* bufferStack, bool leftBuffer, int& curBufferSize, int procRank)
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
		bsEntry->block = curBlockStack->bStack->block;
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
			(bufferStack->bStack)->block.insert((bufferStack->bStack)->block.begin(), curBlockStack->bStack->block.begin(), curBlockStack->bStack->block.end());
		}
		else
		{
			(bufferStack->bStack->next)->block.insert((bufferStack->bStack->next)->block.begin(), curBlockStack->bStack->block.begin(), curBlockStack->bStack->block.end());
			tmpBlockStack->rightBuffer.push_back(curBlockStack);
		}
	}
	curBufferSize = tmpBlockStack->bStack->block.size() + tmpBlockStack->bStack->next->block.size();
	return tmpBlockStack;
} 


TBlockId WLManager::GetBufferData(SPIRITVisitor* vis,TContextVector& lData, TContextVector& rData)
{
	bool flag = false;

	//block contains indices relative to the beginning of the block.
	TIndices::iterator iter = curBlockStack->bStack->block.begin();
	for(;iter!=curBlockStack->bStack->block.end();iter++)
	{
		Point* p = (Point*)(points+(*iter * offsetFactor));
		Context* l=vis->GetContext(p);
		l->index = *iter;

		lData.push_back(l);
	}

	iter = curBlockStack->bStack->next->block.begin();
	for(;iter!=curBlockStack->bStack->next->block.end();iter++)
	{
		Point* p = (Point*)(points+(*iter * offsetFactor));
		Context *l=vis->GetContext(p);
		l->index = *iter;

		rData.push_back(l);
	}
	
	if(lData.size() > 0)
		curBlockStack->numFragments++;
	if(rData.size() > 0)
		curBlockStack->numFragments++;
	return curBlockId;	

}

int WLManager::GetCompressedBlockIDs(TBlockId blockId, Context* lData, TCBSet& blkIdSet)
{
	BlockStack* bufferStack = reinterpret_cast<BlockStack*>(blockId.second);
	bool found = false;

	if(bufferStack->isBuffer)
	{
		bufferStack->numFragments--;
		BSEntry* tmp = bufferStack->bStack;
		for(int i=0;i<tmp->block.size();i++)
		{
			if(lData->index == tmp->block[i])
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

		return STATUS_SUCCESS;
	}

	return STATUS_FAILURE;
}

bool WLManager::IsBufferedBlock(long int blkStack)
{
	BlockStack* bStack = reinterpret_cast<BlockStack*>(blkStack);
	return bStack->isBuffer;
}
#endif


#include <stdio.h>
#include "TriangleKDTreeVisitor.h"
#include <boost/graph/parallel/algorithm.hpp>
typedef ::math::Vector <float, 3> kvec;

bool TriangleKDTreeVisitor::CanCorrelate(GAL_Vertex* node, GAL_Vertex* nextNodeToVisit, TIndices& leftBlock, TIndices& rightBlock)
{
	bool leafNode = node->isLeaf();
	bool correlates = false;

#ifdef TRAVERSAL_PROFILE
		((node))->pointsVisited += block.size();
		((node))->blockOccupancyFactor += block.size()/(double)blockSize;
#endif
	for(int j=0; j<block.size();j++)
	{
		/*if((block[j] == 0) && (node->level==0))
			printf(" debug break\n");*/
		TRayBlockIter piter = rays.begin();
		TInfoBlockIter iiter = infos.begin();
		float time=(float)0.0;
#ifdef TRACK_TRAVERSALS
		((piter+block[j])->num_nodes_traversed)++;
#endif
		if (node->checkRay(*(piter+block[j]), time))
		{
			if(time > (iiter+block[j])->time)
				continue;
			if(!leafNode)
			{
				leftBlock.push_back(block[j]);
				correlates = true;
			}
			else
			{
				bool hit = false;
				time = (float)0.0;
				// Test the triangles in this node for an intersection.
				for (size_t i = 0; i < node->triangles.size(); ++i)
				{
					
					if ((node->triangles[i]).checkRayIntersection((piter+block[j])->origin, (piter+block[j])->direction, time))
					 {

						if (time >= 0.0f) 
						{
							// Ensure that the hitpoint is actually within the bounds of this node.
							// The triangle could get hit outside of this node since triangles
							// can span multiple nodes.
							kvec hit_point = (piter+block[j])->origin + (piter+block[j])->direction * time;
							if (time < (iiter+block[j])->time) 
							{
								hit = true;
								(iiter+block[j])->time		= time;
								(iiter+block[j])->hit_point 	= hit_point;
								(iiter+block[j])->normal		= (node->triangles[i]).interpolateNormal(hit_point, (iiter+block[j])->barycentrics);
								(iiter+block[j])->material 	= (node->triangles[i]).m_material;
								(iiter+block[j])->triangle	= node->triangles[i];
								if ((((iiter+block[j])->material).texture).hasData()) {
									(iiter+block[j])->tex_coord = (node->triangles[i]).interpolateTexture((iiter+block[j])->barycentrics);
								}
							}
						}
					}
				}
				if (hit) 
					(piter+block[j])->intersects = true;

			}
		}
	}

	if(correlates && !leafNode)
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
		block.clear();	
		block = leftBlock;
		stackElm->block = block;
		stackElm->next = curBlockStack->bStack;
		curBlockStack->bStack=stackElm;
	}
	return correlates;

} 

void TriangleKDTreeVisitor::Eff_AddBlockForTraversal(TIndices& pvRef, const GAL_Vertex* startNode, int curBlkId, GAL_Vertex* sib)
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
	block.clear();
	block = pvRef;
	curBlockStack = tmpBlockStack;
	curBlockId = tmpBlockStack->bStackId;

}

int TriangleKDTreeVisitor::Traverse()
{
	int ret;
#ifdef LOAD_BALANCE
	int numProcs = gal->GetOpt_NumProcs();
	workLoad = rays.size()/numProcs;
	//workLoad = rays.size()/4;
#endif
	ret = gal->GAL_Traverse(this);
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

bool TriangleKDTreeVisitor::GALVisitor_VisitNode(GAL_Vertex* node, GAL_Vertex* nextNodeToVisit, TIndices& leftBlock, TIndices& rightBlock, TBlockId& curBlkId)
{
	bool ret = false;
	//std::cout<<process_id((*g).process_group())<<": pot being checked: ("<<(pot).pt[0]<<", "<<(pot).pt[1]<<")"<<std::endl;
	ret = CanCorrelate(node,nextNodeToVisit,leftBlock, rightBlock);
	curBlkId = curBlockId;
	return ret;
		
}

int TriangleKDTreeVisitor::GALVisitor_FillPipeline(GAL_Vertex* startNode)
{
	if(numWorkItems <= 0)
		return STATUS_NO_WORKITEMS;

	if(numPipelineElems == blkIntakeLimit)
	{
		return STATUS_PIPELINE_FULL;
	}

	numPipelineElems++;


	long int startIndx = (numWorkItems-1) * blockSize;
	long int endIndx = (numWorkItems) * blockSize;
#ifdef LOAD_BALANCE
	startIndx = start+startIndx;
	endIndx = start + endIndx;
	if(endIndx > end)
		endIndx = end;
#else
	if(endIndx > rays.size())
		endIndx = rays.size();
#endif
	//printf("startIndx:%d endIndx:%d\n",startIndx, endIndx);
	TIndices tmpBlock(endIndx - startIndx);

	for(long int i=0;i<(endIndx-startIndx);i++)
	{
		tmpBlock[i]=startIndx+i;
	}
#ifdef LOAD_BALANCE
	/*if((start+numWorkItems == 333373) || (start+numWorkItems == 666699))
		printf("Debug.\n");*/
	Eff_AddBlockForTraversal(tmpBlock,startNode, start+numWorkItems, NULL);
#else
	Eff_AddBlockForTraversal(tmpBlock,startNode, numWorkItems, NULL);
#endif
		
	numWorkItems--;
	return STATUS_SUCCESS;
	
}

int TriangleKDTreeVisitor::GALVisitor_GetNumberOfWorkItems()
{
	return bStackListSize;
}


int TriangleKDTreeVisitor::GALVisitor_SetLocalData(const TLocalDataVector& lData, TBlockId origBlkId, bool updateCurBlockStack)
{
	TIndices tmpIndices;

	block.clear();
	
	curBlockId = origBlkId;
		
	TRayBlockIter iter = rays.begin();
	TInfoBlockIter iiter = infos.begin();
	//Aggregate block. Decompress.
	for(int i=0;i<lData.size();i++)
	{
		LocalData l;
#ifdef TRACK_TRAVERSALS
#ifdef STATISTICS2
		(iter +lData[i].index)->numStagesExecuted = lData[i].numStagesExecuted;
		((iter +lData[i].index)->numStagesExecuted)++;
#endif
		(iter +lData[i].index)->num_nodes_traversed = lData[i].nodesTraversed;
#endif
		if(lData[i].intersects)
		{
			(iter +lData[i].index)->intersects = lData[i].intersects;
			*(iiter +lData[i].index) = lData[i].info;
		}
	}

	if(updateCurBlockStack)
	{
		block.clear();
		curBlockId = origBlkId;
			curBlockStack = reinterpret_cast<BlockStack*>(origBlkId.second);
#if 0
		BlockStackListIter tmpIter = bStackList.begin();
		while(tmpIter != bStackList.end())
		{
			long int tmpPRoot = reinterpret_cast<long int>((tmpIter->blkId).first);
			if(((tmpIter->blkId).second == origBlkId.second) && (tmpPRoot == origBlkId.first))
				break;
			tmpIter++;
		}
		
		if(tmpIter == bStackList.end())
			printf("ERROR: BlkId: tmpBlkId %ld\n", origBlkId.second);
		assert(tmpIter != bStackList.end());
		curBlockStack = tmpIter;
#endif
			block = curBlockStack->bStack->block;
	}

	return 0;
}

/* Description: This function should be called only when leaving a process boundary. i.e at pseudoLeaves.
 * returns the locally computed values of 
 * all the elements that are pointed to by the current block indices.
 * Parameters: lData - reference to a vector that contains locally computed data
 * Return Value: blockId of the current block.
 */
TBlockId TriangleKDTreeVisitor::GALVisitor_GetLocalData(TLocalDataVector& lData)
{
	bool flag = false;

	TRayBlockIter iter = rays.begin();
	TInfoBlockIter iiter = infos.begin();

	//block contains indices relative to the beginning of the block.
	for(int i=0;i<block.size();i++)
	{
		LocalData l;
#ifdef TRACK_TRAVERSALS
#ifdef STATISTICS2
		l.numStagesExecuted = (iter +block[i])->numStagesExecuted;
#endif
		l.nodesTraversed = (iter +block[i])->num_nodes_traversed;
#endif
		l.intersects = (iter +block[i])->intersects;
		l.index = block[i];
		l.info = *(iiter+block[i]);
		lData.push_back(l);
	}
	return curBlockId;	
}


void TriangleKDTreeVisitor::GALVisitor_UpdateBlockFromBlockStackTop(GAL_Vertex* searchNode)
{
	if(curBlockStack->bStack)
	{
		block.clear();
		block = curBlockStack->bStack->block;
	}	
}

bool TriangleKDTreeVisitor::GALVisitor_IsLastFragment(GAL_Vertex* node, BlockStackListIter curBStack, TBlockId& parentBlkId)
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


BlockStackListIter TriangleKDTreeVisitor::GALVisitor_GetCurrentBlockStack()
{
	return curBlockStack;
}
void TriangleKDTreeVisitor::GALVisitor_SetAsCurrentBlockStack2(BlockStackListIter curBStack)
{
	curBlockStack = curBStack;
	GALVisitor_UpdateBlockFromBlockStackTop();
	curBlockId = curBlockStack->bStackId;
}

void TriangleKDTreeVisitor::GALVisitor_SetAsCurrentBlockStack(const TBlockId& blockStackId)
{
	curBlockStack = reinterpret_cast<BlockStack*>(blockStackId.second);
	curBlockId = curBlockStack->bStackId;
	GALVisitor_UpdateBlockFromBlockStackTop();
}

int TriangleKDTreeVisitor::GALVisitor_RemoveBlockStack()
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


GAL_Vertex* TriangleKDTreeVisitor::GALVisitor_PopFromCurrentBlockStackAndUpdate()
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

void TriangleKDTreeVisitor::GALVisitor_PushToBlockStackAndUpdate(MsgTraverse& msg)
{
	block.clear();

	{
		TRayBlockIter iter = rays.begin();
		TInfoBlockIter iiter = infos.begin();

		for(int i=0;i<msg.l.size();i++)
		{
			LocalData l;
			block.push_back(msg.l[i].index);
#ifdef TRACK_TRAVERSALS
#ifdef STATISTICS2
			(iter +msg.l[i].index)->numStagesExecuted = msg.l[i].numStagesExecuted;
			((iter +msg.l[i].index)->numStagesExecuted)++;
#endif
			(iter +msg.l[i].index)->num_nodes_traversed = msg.l[i].nodesTraversed;
#endif
			if(msg.l[i].intersects)	
			{
				(iter +msg.l[i].index)->intersects = msg.l[i].intersects;
				*(iiter+msg.l[i].index) = msg.l[i].info;
			}
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

void TriangleKDTreeVisitor::GALVisitor_SetBlock(const TIndices& blk)
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
 * Precondition: 'block' and 'block ID' must already be set.
 * Parameters: pRoot - reference to the pseudo-root
 * Post condition: adds the tuple <block, block ID> to the list maintained by pseudoRoot.
 */
void TriangleKDTreeVisitor::GALVisitor_AddToSuperBlock(GAL_Vertex* pRoot, TLocalDataVector& lData, TBlockId tmpBlockId, GAL_Vertex* nextNodeToVisit, int nextNodeDesc)
{
}

/* Description: This function is called by the Graph Abstraction Layer when the traversal leaves a pseudo-root and resumes at either the parent or sibling.
 * Returns: None
 * Precondition: 'block' and 'block ID' must correspond to the current traversal.
 * Parameters: pRoot - reference to the pseudo-root
 * Post condition: 'block' is set.
 * 		   deletes the tuple <block, block ID> to the list maintained by pseudoRoot.
 */
TBlockId TriangleKDTreeVisitor::GALVisitor_DeleteFromSuperBlock(GAL_Vertex* pRoot, GAL_Vertex** nodeToVisit, int& nextNodeProc)
{
	if(curBlockStack->bStack)
	{
		*nodeToVisit = curBlockStack->bStack->nodeToVisit;
		block = curBlockStack->bStack->block;
	}
	nextNodeProc = curBlockStack->nextNodePid;
	return curBlockStack->parentBlockId;
}

BlockStackListIter TriangleKDTreeVisitor::GALVisitor_CreateBlockStack(const TIndices& blk, const GAL_Vertex* rNode, GAL_Vertex* sibNode, int numExpUpdates, BlockStackListIter& curBStack)
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


void TriangleKDTreeVisitor::GALVisitor_UpdateCurrentBlockStackId(GAL_Vertex* pRoot)
{
}

#ifdef PERFCOUNTERS
BlockStats TriangleKDTreeVisitor::GALVisitor_GetNumberOfBlocksInFlight()
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
#endif

#ifdef LOAD_BALANCE
int TriangleKDTreeVisitor::GALVisitor_GetWorkerId(long int workItemId)
{
	//printf("workItemId:%d workerId:%d\n",workItemId,workItemId/workLoad);
	//workLoad can be zero when there are less number of input points than number of processes.
	if((workLoad == 0) || (workLoad == 1))
		return workItemId-1;
	else
		return workItemId/workLoad;
}
#endif
#ifdef MESSAGE_AGGREGATION
BlockStackListIter TriangleKDTreeVisitor::GALVisitor_CreateBufferStack(BlockStack* bufferStack, bool leftBuffer, int& curBufferSize)
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


TBlockId TriangleKDTreeVisitor::GALVisitor_GetBufferData(TLocalDataVector& lData, TLocalDataVector& rData)
{
	bool flag = false;

	TRayBlockIter iter = rays.begin();
	TInfoBlockIter iiter = infos.begin();
	//block contains indices relative to the beginning of the block.
	for(int i=0;i<block.size();i++)
	{
		LocalData l;
#ifdef TRACK_TRAVERSALS
#ifdef STATISTICS2
		l.numStagesExecuted = (iter +block[i])->numStagesExecuted;
#endif
		l.nodesTraversed = (iter +block[i])->num_nodes_traversed;
#endif
		l.intersects = (iter +block[i])->intersects;
		l.index = block[i];
		l.info = *(iiter+block[i]);
		lData.push_back(l);
	}

	block.clear();
	block = curBlockStack->bStack->next->block;
	for(int i=0;i<block.size();i++)
	{
		LocalData l;
#ifdef TRACK_TRAVERSALS
#ifdef STATISTICS2
		l.numStagesExecuted = (iter +block[i])->numStagesExecuted;
#endif
		l.nodesTraversed = (iter +block[i])->num_nodes_traversed;
#endif
		l.intersects = (iter +block[i])->intersects;
		l.index = block[i];
		l.info = *(iiter+block[i]);
		rData.push_back(l);
	}
	
	if(lData.size() > 0)
		curBlockStack->numFragments++;
	if(rData.size() > 0)
		curBlockStack->numFragments++;
	return curBlockId;	

}

int TriangleKDTreeVisitor::GALVisitor_GetCompressedBlockIDs(TBlockId blockId, LocalData& lData, TCBSet& blkIdSet)
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

		return STATUS_SUCCESS;
	}

	return STATUS_FAILURE;
}

bool TriangleKDTreeVisitor::GALVisitor_IsBufferedBlock(long int blkStack)
{
	BlockStack* bStack = reinterpret_cast<BlockStack*>(blkStack);
	return bStack->isBuffer;
}
#endif


#ifdef LOAD_BALANCE
TBlockId TriangleKDTreeVisitor::GALVisitor_GetCurrentBlockId()
{
	return curBlockId;
}
#endif

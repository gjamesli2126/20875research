#include "SPIRIT.hpp"
#include "WLManager.hpp"
#include "timer.h"
#include<fstream>
#include<sys/mman.h>
#include<sys/stat.h>
#include<fcntl.h>

#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<boost/graph/distributed/adjacency_list.hpp>


SPIRIT* SPIRIT::instance = NULL;
std::ifstream input;
long int labelCount, totalNodesTraversed, traversalsTBD; //counter to assign label values to a node.
long int uniqPLeafLabel; //counter to assign label values to a node.
bool remotePseudoLeavesPresent=false;
std::map<long int,PLeafPointBucket> pLeafMap;

std::vector<Vertex*> rank0PseudoLeaves;
std::vector<float*> rank0PseudoLeavesParams;

Timer busyTime;
#ifdef TRAVERSAL_PROFILE
std::vector<long int> pipelineStage; 
#endif


typedef std::pair<int, int> PBGL_oobMessage;
typedef optional<PBGL_oobMessage> PBGL_AsyncMsg;

/*bool SortPPP(long int a, long int b)
{
	return a>b;
}*/

float mydistance(const Point* a, const Point* b) {
	float d = 0;
	for(int i = 0; i < DIMENSION; i++) {
		float diff = a->pt[i] - b->pt[i];
		d += diff * diff;
	}
	return sqrt(d);
}

struct SplitDimComparator
{
	int split_dim;
	SplitDimComparator(int sd ) : split_dim(sd) {}
	bool operator()(const Point* a, const Point* b) {
		return a->pt[split_dim] < b->pt[split_dim];
	}
	int Compare(const Point* a, const Point* b)
	{
		float lhs = a->pt[split_dim];
		float rhs = b->pt[split_dim];
	
		if(lhs < rhs)
			return -1;
		else if(lhs > rhs)
			return 1;
		else
			return 0;
	}
};


struct DistanceComparator
{
	const Point* item;
	DistanceComparator( const Point* _item ) : item(_item) {}
	bool operator()(const Point* a, const Point* b) {
		return mydistance( item, a ) < mydistance(item, b );
	}
	int Compare(const Point* a, const Point* b)
	{
		float lhs = mydistance(item,a);
		float rhs = mydistance(item,b);
	
		if(lhs < rhs)
			return -1;
		else if(lhs > rhs)
			return 1;
		else
			return 0;
	}
};

/* based on clrs, chap 7,9.*/
int partition(std::vector<Point*>& a, int p, int r, void* comparator, TType type)
{
	Point* x = a[r];
	int i=p-1;
	if(type == VPTREE)
	{	
		for(int j=p;j<=r-1;j++)
		{
			if(((DistanceComparator*)comparator)->Compare(a[j], x) == -1)
			{
				i=i+1;
				std::swap(a[i],a[j]);
			}
		}
	}
	else if((type == KDTREE)||(type == PCKDTREE))
	{
		for(int j=p;j<=r-1;j++)
		{
			if(((SplitDimComparator*)comparator)->Compare(a[j], x) == -1)
			{
				i=i+1;
				std::swap(a[i],a[j]);
			}
		}
	}
	std::swap(a[i+1],a[r]);
	return i+1;
}

void find_median(std::vector<Point*>& a, int p, int r, int i, void* comparator, TType type)
{
	if((p==r)||(p>r))
		return;
	int q = partition(a, p, r, comparator, type);
	int k = q-p+1;
	if(k==i)
		return; 
	else if(i<k)
		return find_median(a,p,q-1,i, comparator, type);
	else
		return find_median(a,q+1,r,i-k, comparator, type);
			
}

void my_nth_element(std::vector<Point*>& points, int from, int mid, int to,void* comparator, TType type)
{
	find_median(points, from, to, mid, comparator, type); 
}

Vertex::~Vertex()
{
	VertexData* tmpvData = vData;
	while(tmpvData)
	{
		VertexData* tmpvData2 = tmpvData->next;
		delete tmpvData;
		tmpvData = tmpvData2;
	}
}

void Vertex::GetPLeafChildrenDetails(Vertex* nextNodeToVisit, MsgTraverse& msgTraverse)
{
	bool flag = false;
	for(int i=0;i<8;i++)
	{
		if(pChild[i] && flag)
		{
			msgTraverse.pSiblings.push_back(reinterpret_cast<long int>(pChild[i]));
			msgTraverse.siblingDescs.push_back(childDesc[i]);
		}
		if(pChild[i] && (pChild[i] == nextNodeToVisit))
		{
			flag = true;
		}
	}
}

Vertex* SPIRIT::CreateVertex(TVertexType vType)
{
	labelCount++;
	Vertex* node; 
	if(vType == VTYPE_PSEUDOROOT)
	{
		node = new SPIRIT_PseudoRoot();
#ifdef TRAVERSAL_PROFILE
		pipelineStage.push_back(reinterpret_cast<long int>(node));
#endif
	}
	else
	{
		node =  new Vertex();
	}
	node->label = labelCount;
	return node;
}

void SPIRIT::SendMessage(int processId, int messageId, void* msg)
{
	busyTime.Stop();
	assert(processId != procRank);
	assert(0<=processId<numProcs);
	if(processId == procRank)
		printf("ERROR same process %d\n",procRank);
	switch(messageId)
	{
		case MESSAGE_BUILDSUBTREE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgBuildSubTree *>(msg))); 
					break;
		case MESSAGE_DONEOCTSUBTREE_ACK:
					send_oob(pg,processId,messageId,*(reinterpret_cast<bool *>(msg))); 
					break;
		case MESSAGE_DONETREE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<int *>(msg))); 
					break;
		case MESSAGE_TRAVERSE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgTraverse*>(msg))); 
					break;
		case MESSAGE_TRAVERSE_BACKWARD:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgTraverse*>(msg))); 
					break;
		case MESSAGE_UPDATE_PLEAVES_OCT:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgUpdatePLeaves_Oct*>(msg))); 
					break;
		case MESSAGE_READYTOEXIT:
					send_oob(pg,processId,messageId,*(reinterpret_cast<int*>(msg))); 
					break;
		case MESSAGE_REPLICATE_REQ:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgReplicateReq*>(msg))); 
					break;
		default:
			break;
	}
}


void SPIRIT::PrintGraph()
{
#ifdef STATISTICS
	long int totalTraverseCount;
	reduce(communicator(pg),traverseCount, totalTraverseCount, std::plus<long int>(),0);
	float localPPM=0., avgPPM;
	if(pointsProcessed != 0)
		localPPM = pointsProcessed/(float)traverseCount;
		
	reduce(communicator(pg),localPPM, avgPPM, std::plus<float>(),0);
	if(procRank==0)
		printf("Total Messages Processed:%d  Avg Points/Message:%f\n",totalTraverseCount, avgPPM/numProcs);

	long int totalPointsProcessed;
	reduce(communicator(pg),pointsProcessed, totalPointsProcessed, std::plus<long int>(),0);
	if(procRank==0)
		printf("TotalPointsProcessed: %ld\n",totalPointsProcessed);
	
#ifdef MESSAGE_AGGREGATION
		long int totalBufferCount=0;
		reduce(communicator(pg),bufferStage, totalBufferCount, std::plus<long int>(),0);
		if(procRank == 0)
			printf("Total Points Buffered: %ld\n",totalBufferCount);
#endif
#endif
#ifdef TRAVERSAL_PROFILE
	long int localPointsVisited=0, totalPointsVisited;
	SPIRITHelper_GetTotalPointsVisited(rootNode, localPointsVisited);
	reduce(communicator(pg),localPointsVisited, totalPointsVisited, std::plus<long int>(),0);
	double comm_comp_ratio=traverseCount/(double)(localPointsVisited);
	double avgRatio;
	reduce(communicator(pg),comm_comp_ratio, avgRatio, std::plus<double>(),0);
	if(procRank == 0)
		printf("Average communication to computation ratio: %f\n",avgRatio/numProcs);
	//printf("localPointsVisited: %ld traverseCount: %ld\n",localPointsVisited, traverseCount);
#if 0
	//uncomment this part for collecting per-subtree load
	if(procRank == 0)
	{
		std::vector<long int>::iterator iter = pipelineStage.begin();
		for(;iter!=pipelineStage.end();iter++)
		{
			Vertex* pRoot = reinterpret_cast<Vertex*>(*iter);
			if((pRoot->level != 0) && (pRoot->leftChild || pRoot->rightChild))
			{
				long int count=0;
				SPIRITHelper_CountSubtreeNodes(pRoot, count);
				printf("Subtree %ld Points_Visited %ld\n",*iter,count);
			}
		}
	}
#endif
#endif


}

int SPIRIT::SPIRITHelper_HandleMessageTraverseBackward(Vertex* childNode, TBlockId curBlockId)
{
	int ret = STATUS_TRAVERSE_INCOMPLETE;
	Vertex* tmpChildNode = childNode;
	Vertex* nextNodeToVisit = NULL;

	if(curTreeType == OCTREE)
		nextNodeToVisit  = wlManager->PopFromCurrentBlockStackAndUpdate();

	while(true)
	{
		TBlockId fragBlkId;
		fragBlkId.second = INVALID_BLOCK_ID; 
		BlockStackListElem curBStack = wlManager->GetCurrentBlockStack(); 
		bool readyToSend = wlManager->IsLastFragment(tmpChildNode, curBStack, fragBlkId);	
		if(fragBlkId.second != INVALID_BLOCK_ID)
		{
			wlManager->PopFromCurrentBlockStackAndUpdate();
			wlManager->RemoveBlockStack(vis);
			if(!readyToSend)
			{
				break;
			}
			else
			{	
				curBlockId = fragBlkId;
#ifdef MESSAGE_AGGREGATION
				if(tmpChildNode->pseudoRoot)
				{
					fragBlkId.second = INVALID_BLOCK_ID; 
					bool readyToSend = wlManager->IsLastFragment(tmpChildNode, reinterpret_cast<BlockStack*>(curBlockId.second), fragBlkId);	
					if(fragBlkId.second != INVALID_BLOCK_ID)
					{
						wlManager->SetAsCurrentBlockStack2(reinterpret_cast<BlockStack*>(curBlockId.second));
						wlManager->PopFromCurrentBlockStackAndUpdate();
						wlManager->RemoveBlockStack(vis);
						if(!readyToSend)
						{
							break;
						}
						else
						{	
							curBlockId = fragBlkId;
						}
					}
				}
#endif
				wlManager->SetAsCurrentBlockStack(curBlockId);
			}
		}
			
		long int parent = reinterpret_cast<long int>(tmpChildNode->parent);
		int procId = (tmpChildNode->parentDesc);
		if(tmpChildNode->pseudoRoot && (tmpChildNode->level == opts.subtreeHeight))
		{
			curBStack = reinterpret_cast<BlockStack*>(curBlockId.second);
			procId = curBStack->parentBlockId.first;
			assert(procId < numProcs);
			parent = ((SPIRIT_PseudoRoot*)(tmpChildNode))->parents2[procId];
			assert(parent != 0);
			if(procId == 0)
				parent = reinterpret_cast<long int>(tmpChildNode->parent);
		}
		if(curTreeType != OCTREE)
			nextNodeToVisit  = wlManager->PopFromCurrentBlockStackAndUpdate();
		else
			nextNodeToVisit  = wlManager->GetNextNodeToVisit(tmpChildNode, true);
			

		if(nextNodeToVisit != NULL)
		{
			tmpChildNode = nextNodeToVisit;
			if(curTreeType != OCTREE)
				nextNodeToVisit = NULL;	
			else
				nextNodeToVisit = wlManager->GetNextNodeToVisit(tmpChildNode, false);
			busyTime.Start();
			ret = SPIRIT_TraverseHelper(tmpChildNode, nextNodeToVisit);
			busyTime.Stop();
			if(ret != STATUS_TRAVERSE_COMPLETE)
			{
				break;
			}
			else
			{
				if(nextNodeToVisit)
					continue;
				else if(curTreeType == OCTREE)
					wlManager->GetNextNodeToVisit(tmpChildNode, true);
			}
			//parent of tmpChildNode and NextNode are same.
		}

		if(tmpChildNode->parent == NULL) 
		{
			ret = 1;
			break;
		}
		
		tmpChildNode = reinterpret_cast<Vertex*>(parent);//tmpChildNode->parent;
	}
	return ret;	
	
}



int SPIRIT::SPIRIT_TraverseHelper_SendBlocks()
{
	int ret =STATUS_TRAVERSE_INCOMPLETE;
	int status = STATUS_SUCCESS;
	do
	{
		status = wlManager->FillPipeline(rootNode, procRank);
		if((status == STATUS_NO_WORKITEMS) ||(status == STATUS_PIPELINE_FULL)) 
		{
			if((status == STATUS_NO_WORKITEMS) && (wlManager->GetNumberOfWorkItems() == 0))
				ret=STATUS_TRAVERSE_COMPLETE;
			else
			{
				ret=STATUS_TRAVERSE_INCOMPLETE;
#ifdef MESSAGE_AGGREGATION
				if(!readyToFlush)
				{
					readyToFlush = true;
				}
#endif
			}

		}
		else
		{
			busyTime.Start();
			int ret = SPIRIT_TraverseHelper(rootNode, NULL);
			busyTime.Stop();
			
			if(ret == STATUS_TRAVERSE_INCOMPLETE)
			{
				status = STATUS_FAILURE;
			}
			else if(ret == STATUS_TRAVERSE_COMPLETE)
			{
				int re = wlManager->RemoveBlockStack(vis);		
			}

		}
	}
	while(status == STATUS_SUCCESS);

#ifdef MESSAGE_AGGREGATION
	if(readyToFlush)
	{
		if(aggrBuffer.size() > 0)
		{
			SPIRIT_TraverseHelper_SendCompressedMessages();
		}
	}
#endif

	return ret;
}

void SPIRIT::BlockCyclicDistribution(long int travTBD, int subBlockSize)
{
	long int startIndex, endIndex;
	if((numProcs*subBlockSize) > travTBD)
	{
		startIndex = procRank * travTBD/numProcs;
		endIndex = (procRank == numProcs-1)?(travTBD):((procRank+1) * travTBD/numProcs);
		wlManager->start=startIndex;
		wlManager->end=endIndex;
		opts.distribution=BLOCK_DIST;
		wlManager->InitializeState();
		return;
	}

	startIndex = procRank*subBlockSize;
	endIndex = startIndex+subBlockSize;
	if(endIndex > travTBD)
		endIndex = travTBD;
	wlManager->AddSubBlock(startIndex, endIndex);
	while((startIndex < travTBD) && (endIndex <= travTBD))
	{ 
		startIndex += numProcs*subBlockSize;
		if(startIndex < travTBD)
		{
			endIndex = startIndex+subBlockSize;
			if(endIndex > travTBD)
				endIndex = travTBD;
			wlManager->AddSubBlock(startIndex, endIndex);
		}
	}
	wlManager->PrepareWorkBlocks();
	return;
}


int SPIRIT::Traverse(SPIRITVisitor* v)
{
	PBGL_oobMessage msg;
	bool done = false;
	int ret = STATUS_TRAVERSE_INCOMPLETE;
	double stepTime=0, startTime, endTime, traversalTime=0.;
	vis = v;
	int offsetFactor=0;
	if(curTreeType == OCTREE)
		offsetFactor = sizeof(OctreePoint);
	else if(curTreeType == KDTREE)
		offsetFactor = sizeof(KdtreePoint);
	else if(curTreeType == PCKDTREE)
		offsetFactor = sizeof(PCKdtreePoint);
	else if(curTreeType == VPTREE)
		offsetFactor = sizeof(VptreePoint);

	int subBlockSize = opts.blockSize/8;

	long int curTraversals=0;
	if((parser->GetTreeType() == KDTREE) ||(parser->GetTreeType() == VPTREE))
		curTraversals = numTraversals/2; //skip the training set.

	int step=0;
	long int startIndex, endIndex;
	while(curTraversals < numTraversals)	
	{
		synchronize(pg);
		step++;
		if((parser->GetTreeType() == OCTREE) || (step!=1))
		{	
			if(numTraversals > opts.batchSize)
			{
				unsigned char* p = address;
				//if(procRank < 10)
					traversalsTBD = ReadTreeData_Oct(NULL, curTraversals, numTraversals, p, parser, step);
				//broadcast(communicator(pg), traversalsTBD, 0);
			}
			else
			{
				traversalsTBD  = numTraversals - curTraversals;
			}
		}

		if(procRank == 0)
			printf("Traversals done %ld (TBD %ld) TotalTraversals %ld (steptime %f) \n",curTraversals, traversalsTBD, numTraversals, stepTime);
		startTime = clock();
		curTraversals += traversalsTBD;
		if(opts.distribution == BLOCK_DIST)
		{
			
			//block distribution
			if(opts.pipelined)
			{
				if(procRank == 0)
				{
					startIndex = 0;
					endIndex = traversalsTBD;
				}
				else
					startIndex = endIndex = 0;
			}
			else
			{
				startIndex = procRank * traversalsTBD/numProcs;
				endIndex = (procRank == numProcs-1)?(traversalsTBD):((procRank+1) * traversalsTBD/numProcs);
			}
			if(!wlManager)
				wlManager = new WLManager(opts, address, traversalsTBD, startIndex, endIndex, offsetFactor);
			else
			{
				wlManager->numPoints = traversalsTBD;
				wlManager->start = startIndex;
				wlManager->end = endIndex;
				wlManager->InitializeState();
			}
			
		}
		else if(opts.distribution == BLOCK_CYCLIC)
		{
			//Block-cyclic distribution
			if(!wlManager)
				wlManager = new WLManager(opts, address, traversalsTBD, offsetFactor);
			else
				wlManager->numPoints = traversalsTBD;
			BlockCyclicDistribution(traversalsTBD, subBlockSize);
		}
		
		synchronize(pg);
		while (!done)
		{
			//poll for messages
			PBGL_AsyncMsg pollMsg = pg.poll();
			if(!pollMsg)
			{
				if(vis)
				{
					ret = SPIRIT_TraverseHelper_SendBlocks();
					if(ret == STATUS_TRAVERSE_COMPLETE)
					{
						if(!readyToExit)
						{
							readyToExitList.insert(procRank);
							if(readyToExitList.size() != numProcs)
							{
								for(int i=0;i<numProcs;i++)
								{
									if(i !=procRank)
										SendMessage(i,MESSAGE_READYTOEXIT,&procRank); 
								}
								readyToExit = true;
							}
							else
							{
								int ter;
								for(int i=0;i<numProcs;i++)
								{
									if(i !=procRank)
										SendMessage(i,MESSAGE_DONETREE,&ter); 
								}
								done = true;
								break;
							}
						}
					}
				}
				continue;
			}
			else
			{
				msg = pollMsg.get();

				//poll for messages
				switch(msg.second)
				{
					case MESSAGE_TRAVERSE:
					{
							MsgTraverse msgTraverse;
							//receive_oob(pg, msg->first, msg->second, msgTraverse);
							receive_oob(pg, msg.first, msg.second, msgTraverse);
							Vertex* pRoot = reinterpret_cast<Vertex*>(msgTraverse.pRoot);
							SPIRIT_PseudoRoot* pSibling = reinterpret_cast<SPIRIT_PseudoRoot*>(msgTraverse.pSibling);
							bool loadBalanced = false;
							int procId = msgTraverse.blkStart.first;
							if(procId == procRank)
							{
								loadBalanced = true;
							}
							if(!pRoot->leaf)
								((SPIRIT_PseudoRoot*)pRoot)->parents2[procId]=msgTraverse.pLeaf;
	#ifdef STATISTICS
							traverseCount++;
							pointsProcessed += msgTraverse.l.size();
	#endif
							if(!loadBalanced)
								wlManager->PushToBlockStackAndUpdate(vis, msgTraverse, procRank);
							else
								wlManager->SetContext(vis,msgTraverse.l,msgTraverse.blkStart);

							ret = SPIRITHelper_DeAggregateBlockAndSend(&pRoot, msgTraverse, loadBalanced);

							if(ret == STATUS_TRAVERSE_COMPLETE)
							{
								if(!loadBalanced)
								{
									wlManager->RemoveBlockStack(vis);	
									SendMessage(procId,MESSAGE_TRAVERSE_BACKWARD, &msgTraverse);
									ret = STATUS_TRAVERSE_INCOMPLETE;
								}
								else
								{
									bool decodeBlocks = false;
	#ifdef MESSAGE_AGGREGATION
									decodeBlocks = SPIRITHelper_HandleMessageTraverseBackward_Multiple(pRoot->parent, msgTraverse, MESSAGE_TRAVERSE);
	#endif
									if(!decodeBlocks)
									{
										wlManager->SetAsCurrentBlockStack(msgTraverse.blkStart);
										Vertex* pLeaf = reinterpret_cast<Vertex*>(msgTraverse.pLeaf);
										int status = SPIRITHelper_HandleMessageTraverseBackward(pLeaf, msgTraverse.blkStart);
										if(status == 1)
										{
											int re = wlManager->RemoveBlockStack(vis);		
										}
									}
								}
							}
								
					}
					break;
					case MESSAGE_TRAVERSE_BACKWARD:
					{
							MsgTraverse msgBkTraverse;
							int status;
							receive_oob(pg, msg.first, msg.second, msgBkTraverse);
							Vertex* parentNode = reinterpret_cast<Vertex*>(msgBkTraverse.pLeaf);
							Vertex* childNode = reinterpret_cast<Vertex*>(msgBkTraverse.pRoot);
							bool decodeBlocks = false;
	#ifdef MESSAGE_AGGREGATION
							TCBSet blkIdSet;
							decodeBlocks = SPIRITHelper_HandleMessageTraverseBackward_Multiple(parentNode, msgBkTraverse, MESSAGE_TRAVERSE_BACKWARD);
							if(!decodeBlocks)
	#endif
							{
								bool waitForSibling = false;
								int re = wlManager->SetContext(vis,msgBkTraverse.l, msgBkTraverse.blkStart);
								status = SPIRITHelper_HandleMessageTraverseBackward(parentNode, msgBkTraverse.blkStart);
								if(status == 1)
								{
										int re = wlManager->RemoveBlockStack(vis);		
								}
							}
							
					}
					break;
					case MESSAGE_DONETREE:
					{
						int ter;
						receive_oob(pg, msg.first, msg.second, ter);
						done = true;
					}
					break;
					case MESSAGE_READYTOEXIT:
					{
						int doneProcId;
						receive_oob(pg, msg.first, msg.second, doneProcId);
						readyToExitList.insert(doneProcId);
						if(readyToExitList.size() == numProcs)
						{
							done = true;
						}
					}
					break;

					default: break;
				}
			}
		}
		readyToExitList.clear();
		readyToExit=false;
		done = false;
		endTime = clock();
		stepTime = (endTime-startTime)/CLOCKS_PER_SEC; 
		traversalTime += stepTime;
		AggregateResults();
	}
	input.close();

	FreeWorkspace();

	double maxTraversalTime;
	reduce(communicator(pg),traversalTime, maxTraversalTime, boost::parallel::maximum<double>(),0);
	printf("Traversal time %f busyTime %f\n",traversalTime, busyTime.GetTotTime()/(double)CLOCKS_PER_SEC);
	if(procRank == 0)
		printf("Traversal time: %f seconds\n",maxTraversalTime);
	return ret;
}


int SPIRIT::SPIRIT_TraverseHelper(Vertex* node, Vertex* sib)
{
		TIndices leftBlock, rightBlock, superBlock;
		int ret = STATUS_TRAVERSE_COMPLETE;
		TBlockId blockId;
		if(!node)
			return STATUS_TRAVERSE_COMPLETE;
		BlockStackListElem rightBStack, curBStack = wlManager->GetCurrentBlockStack();
		
		bool traverseSubTree = wlManager->VisitNode(node, vis, sib, leftBlock, rightBlock, blockId);
		if(!traverseSubTree)
		{
			return STATUS_TRAVERSE_COMPLETE;
		}


		if(node->leaf)
		{
			return STATUS_TRAVERSE_COMPLETE;
		}
		
		
		int expectedUpdates = 2 * ((leftBlock.size()>0) + (rightBlock.size()>0));
		
		if(expectedUpdates > 2)
		{
			rightBStack = wlManager->CreateBlockStack(rightBlock, node, NULL, 2, curBStack, procRank);
			wlManager->CreateBlockStack(leftBlock, node, NULL, 2, curBStack, procRank);
		}
	
		//traverse left subtree
		if(leftBlock.size() > 0)
		{
			if(curTreeType != OCTREE)
			{
				leftBlock.erase(leftBlock.begin(),leftBlock.end());
				if(node->leftChild && ((node->leftDesc) != procRank))
				{	
					MsgTraverse msgTraverse, msgTraverseRight;
					msgTraverse.type  = curTreeType;
					msgTraverseRight.type  = curTreeType;
					blockId = wlManager->GetContext(vis,msgTraverse.l);
					msgTraverse.pRoot = reinterpret_cast<long int>(node->leftChild);
					//msgTraverse.pLeafThreshold = node->vData.threshold;
					msgTraverse.pLeaf = reinterpret_cast<long int>(node);
					msgTraverse.blkStart = blockId;
					if(node->rightChild)
					{
						msgTraverse.pSibling = reinterpret_cast<long int>(node->rightChild);
						msgTraverse.siblingDesc = node->rightDesc;
					}
					else
					{
						msgTraverse.pSibling = static_cast<long int>(0);
					}
	#ifdef MESSAGE_AGGREGATION
					int aggrBlockSize = opts.pipelineBufferSize;
					int bufCount=opts.blockSize;

					if(opts.pipelineBufferSize > 1)	
					{
						bufCount = SPIRIT_TraverseHelper_CompressMessages(msgTraverse, aggrBlockSize, msgTraverseRight,true);
#ifdef STATISTICS
						if(bufCount>msgTraverse.l.size())
							bufferStage += msgTraverse.l.size();
#endif
					}

					if(bufCount >= aggrBlockSize)
					{
	#endif
					SendMessage((node->leftDesc),MESSAGE_TRAVERSE, &msgTraverse);
					ret = STATUS_TRAVERSE_INCOMPLETE;
	#ifdef MESSAGE_AGGREGATION
						if(msgTraverseRight.l.size() > 0)
						{
							//msgTraverseRight.pLeafThreshold = node->vData.threshold;
							msgTraverseRight.pLeaf = reinterpret_cast<long int>(node);
							msgTraverseRight.pRoot = reinterpret_cast<long int>(node->rightChild);
							if(node->leftChild)
							{
								msgTraverseRight.pSibling = reinterpret_cast<long int>(node->leftChild);
								msgTraverseRight.siblingDesc = node->leftDesc;
							}
							else
							{
								msgTraverseRight.pSibling = static_cast<long int>(0);
							}
							SendMessage((node->rightDesc),MESSAGE_TRAVERSE, &msgTraverseRight);
							ret = STATUS_TRAVERSE_INCOMPLETE;
						}
				
					}
					else
					{
						ret = STATUS_TRAVERSE_INCOMPLETE;
					}
	#endif
				}
				else
				{
					if(node->leftChild)
					{
						if((node->leftChild->level==opts.subtreeHeight) && (node->leftChild->leftChild || node->leftChild->rightChild))
						{
							SPIRIT_PseudoRoot* pRoot = (SPIRIT_PseudoRoot*)(node->leftChild);
							pRoot->parents2[procRank] = reinterpret_cast<long int>(node);
						}
						ret = SPIRIT_TraverseHelper(node->leftChild, node->rightChild);
					}
					if(ret == STATUS_TRAVERSE_COMPLETE)
					{
						if(node->rightChild && ((node->rightDesc) != procRank))
						{	
							MsgTraverse msgTraverse;
							msgTraverse.type  = curTreeType;
							blockId = wlManager->GetContext(vis,msgTraverse.l);
							msgTraverse.blkStart = blockId;
							msgTraverse.pRoot = reinterpret_cast<long int>(node->rightChild);
							//msgTraverse.pLeafThreshold = node->vData.threshold;
							msgTraverse.pLeaf = reinterpret_cast<long int>(node);
							msgTraverse.pSibling = static_cast<long int>(0);
							msgTraverse.siblingDesc = node->rightDesc;
							SendMessage((node->rightDesc),MESSAGE_TRAVERSE, &msgTraverse);
							ret = STATUS_TRAVERSE_INCOMPLETE;
						}
						else
						{
							if(node->rightChild)
							{
								if((node->rightChild->level==opts.subtreeHeight) && (node->rightChild->leftChild || node->rightChild->rightChild))
								{
									SPIRIT_PseudoRoot* pRoot = (SPIRIT_PseudoRoot*)(node->rightChild);
									pRoot->parents2[procRank] = reinterpret_cast<long int>(node);
								}
								ret = SPIRIT_TraverseHelper(node->rightChild, NULL);
							}
							if(ret == STATUS_TRAVERSE_COMPLETE)			
							{
								wlManager->PopFromCurrentBlockStackAndUpdate();
								if(expectedUpdates > 2)
								{
									int re = wlManager->RemoveBlockStack(vis);
									if(re == -1)
										printf("Debug 9\n");
									assert(curBStack->numFragments > 0);
								}
							}	
						}
					}
						
				}
			}
			else
			{
				for(int i=0; i<8;i++)
				{
					if(!node->pChild[i])
						continue;
					
					int siblingDesc;
					Vertex* rightSibling = NULL;
					if((node->childDesc[i]) != procRank)
					{	
						MsgTraverse msgTraverse, msgTraverseRight;
						msgTraverse.type  = curTreeType;
						blockId = wlManager->GetContext(vis,msgTraverse.l);
						msgTraverse.blkStart = blockId;
						msgTraverse.pRoot = reinterpret_cast<long int>(node->pChild[i]);
						msgTraverse.pLeaf = reinterpret_cast<long int>(node);
						node->GetPLeafChildrenDetails(node->pChild[i], msgTraverse);
						
				#ifdef MESSAGE_AGGREGATION
						int aggrBlockSize = opts.pipelineBufferSize;
						int bufCount=opts.blockSize;

						if(opts.pipelineBufferSize > 1)	
						{
							bufCount = SPIRIT_TraverseHelper_CompressMessages(msgTraverse, aggrBlockSize, msgTraverseRight,true);
#ifdef STATISTICS
							if(bufCount>msgTraverse.l.size())
								bufferStage += msgTraverse.l.size();
#endif
						}
						assert(msgTraverseRight.l.size() == 0);

						if(bufCount >= aggrBlockSize)
				#endif
						{
							SendMessage((node->childDesc[i]),MESSAGE_TRAVERSE, &msgTraverse);
							ret = STATUS_TRAVERSE_INCOMPLETE;
						}
				#ifdef MESSAGE_AGGREGATION
						else
						{
							ret = STATUS_TRAVERSE_INCOMPLETE;
						}
				#ifdef STATISTICS
						if(bufCount>=aggrBlockSize)
							bufferStage += bufCount;
				#endif
				#endif
						break;
					}
					else
					{
						
						if((numProcs > 1) && (node->pChild[i]->level == opts.subtreeHeight) && !(node->pChild[i]->leaf))
						{
							SPIRIT_PseudoRoot* pRoot = (SPIRIT_PseudoRoot*)(node->pChild[i]);
							pRoot->parents2[procRank] = reinterpret_cast<long int>(node);
						}
						ret = SPIRIT_TraverseHelper(node->pChild[i], rightSibling);
						if(ret == STATUS_TRAVERSE_COMPLETE)
						{
							continue;
						}
						else
							break;
							
					}
				}
				if(ret == STATUS_TRAVERSE_COMPLETE)			
				{
					wlManager->PopFromCurrentBlockStackAndUpdate();
				}
			}	
		}
		

		if(expectedUpdates > 2)
		{
			assert(rightBlock.size() > 0);
			wlManager->SetAsCurrentBlockStack2(rightBStack);
		}
		
		int ret2 = STATUS_TRAVERSE_COMPLETE;
		if(rightBlock.size() > 0)
		{
			rightBlock.erase(rightBlock.begin(),rightBlock.end());
			//first traverse right subtree
			if(node->rightChild && ((node->rightDesc) != procRank))
			{	
				MsgTraverse msgTraverse, msgTraverseRight;
				msgTraverse.type  = curTreeType;
				msgTraverseRight.type  = curTreeType;
				blockId = wlManager->GetContext(vis,msgTraverseRight.l);
				msgTraverseRight.pRoot = reinterpret_cast<long int>(node->rightChild);
				msgTraverseRight.pLeaf = reinterpret_cast<long int>(node);
				msgTraverseRight.blkStart = blockId;
				if(node->leftChild)
				{
					msgTraverseRight.pSibling = reinterpret_cast<long int>(node->leftChild);
					msgTraverseRight.siblingDesc = node->leftDesc;
				}
				else
				{
					msgTraverseRight.pSibling = static_cast<long int>(0);
				}
				
		#ifdef MESSAGE_AGGREGATION
				int aggrBlockSize = opts.pipelineBufferSize;
				int bufCount=opts.blockSize;

				if(opts.pipelineBufferSize > 1)	
				{
					bufCount = SPIRIT_TraverseHelper_CompressMessages(msgTraverse, aggrBlockSize, msgTraverseRight,false);
#ifdef STATISTICS
					if(bufCount>msgTraverseRight.l.size())
						bufferStage += msgTraverseRight.l.size();
#endif
				}

				if(bufCount >= aggrBlockSize)
				{
#endif
				SendMessage((node->rightDesc),MESSAGE_TRAVERSE, &msgTraverseRight);
				ret2 = STATUS_TRAVERSE_INCOMPLETE;
#ifdef MESSAGE_AGGREGATION
					if(msgTraverse.l.size() > 0)
					{
						msgTraverse.pLeaf = reinterpret_cast<long int>(node);
						msgTraverse.pRoot = reinterpret_cast<long int>(node->leftChild);
						if(node->rightChild)
						{
							msgTraverse.pSibling = reinterpret_cast<long int>(node->rightChild);
							msgTraverse.siblingDesc = node->rightDesc;
						}
						else
						{
							msgTraverse.pSibling = static_cast<long int>(0);
						}
						SendMessage((node->leftDesc),MESSAGE_TRAVERSE, &msgTraverse);
						ret = STATUS_TRAVERSE_INCOMPLETE;
					}
				}
				else
				{
					ret2 = STATUS_TRAVERSE_INCOMPLETE;
				}
#endif
			}
			else
			{
				if(node->rightChild)
				{
					if((node->rightChild->level==opts.subtreeHeight) && (node->rightChild->leftChild || node->rightChild->rightChild))
					{
						SPIRIT_PseudoRoot* pRoot = (SPIRIT_PseudoRoot*)(node->rightChild);
						pRoot->parents2[procRank] = reinterpret_cast<long int>(node);
					}
					ret2 = SPIRIT_TraverseHelper(node->rightChild, node->leftChild);
				}
				if(ret2 == STATUS_TRAVERSE_COMPLETE)
				{
					if(node->leftChild && ((node->leftDesc) != procRank))
					{	
						MsgTraverse msgTraverse;
						msgTraverse.type  = curTreeType;
						blockId = wlManager->GetContext(vis,msgTraverse.l);
						msgTraverse.blkStart = blockId;
						msgTraverse.pRoot = reinterpret_cast<long int>(node->leftChild);
						msgTraverse.pLeaf = reinterpret_cast<long int>(node);
						msgTraverse.pSibling = static_cast<long int>(0);
						msgTraverse.siblingDesc = node->leftDesc;
						SendMessage((node->leftDesc),MESSAGE_TRAVERSE, &msgTraverse);
						ret2 = STATUS_TRAVERSE_INCOMPLETE;
					}
					else
					{
						if(node->leftChild)
						{
							if((node->leftChild->level==opts.subtreeHeight) && (node->leftChild->leftChild || node->leftChild->rightChild))
							{
								SPIRIT_PseudoRoot* pRoot = (SPIRIT_PseudoRoot*)(node->leftChild);
								pRoot->parents2[procRank] = reinterpret_cast<long int>(node);
							}
							ret2 = SPIRIT_TraverseHelper(node->leftChild, NULL);
						}
						if(ret2 == STATUS_TRAVERSE_COMPLETE)			
						{
							wlManager->PopFromCurrentBlockStackAndUpdate();
							if(expectedUpdates > 2)
							{
								int re = wlManager->RemoveBlockStack(vis);
								if(re == -1)
									printf("Debug 10\n");
								wlManager->SetAsCurrentBlockStack2(curBStack);
							}
						}
					}
				}
					
			}
				
		}

		if((ret2 == STATUS_TRAVERSE_COMPLETE) && (ret == STATUS_TRAVERSE_COMPLETE))		
		{
			if(expectedUpdates > 2)
				wlManager->PopFromCurrentBlockStackAndUpdate();
			ret  = STATUS_TRAVERSE_COMPLETE;
		}
		else
		{
			ret = STATUS_TRAVERSE_INCOMPLETE;
		}
	
	return ret;
}

#ifdef MESSAGE_AGGREGATION
/* Description: This function is called at a pseudo leaf to compress messages.
 * Parameters: msgTraverse - INOUT - reference to a MsgTraverse. It contains the aggregated vector if the buffer length exceeds PIPELINE_BUFFER_SIZE. Left untouched otherwise.
 * 		blkSTart - IN - block ID of the block relative to which the pointers present in msgTraverse.l are valid.
 * 		node - IN - pseudo leaf node whose left child needs to be visited.
 * 		goingLeft IN - - is true when the block is supposed to be sent to leftChild. False when the bock is to be sent to right Child.
 * 		msgTraverseRight - OUT - contains valid data if the pipelined buffer overflows at this node and there are some blocks that are to be sent right (if going left is false).
 * Return Value: size of the aggregated buffer at this node.
 */
int SPIRIT::SPIRIT_TraverseHelper_CompressMessages(MsgTraverse& msgTraverse, int aggrBlockSize, MsgTraverse& msgTraverseRight, bool goingLeft)
{
	TBlockId blockId;
	TIndices tmpIndices;
	long int pseudoLeaf;
	if(goingLeft)
		pseudoLeaf = msgTraverse.pLeaf;
	else
		pseudoLeaf = msgTraverseRight.pLeaf;
		
	BlockStack* bufferStack;
	int aggrSize=0;
	//Get the current buffer at the vertex
	std::map<long int, long int>::iterator bufIter = aggrBuffer.begin();
	while(bufIter != aggrBuffer.end())
	{
		if(bufIter->first == pseudoLeaf)
		{
			bufferStack = wlManager->CreateBufferStack(reinterpret_cast<BlockStack*>(bufIter->second),goingLeft,aggrSize, procRank);
			break;
		}
		bufIter++;
	}
	//create an entry in the buffer for this 'node' is not found
	if(bufIter == aggrBuffer.end())
	{
		bufferStack = wlManager->CreateBufferStack(NULL,goingLeft,aggrSize, procRank);
		std::pair<std::map<long int, long int>::iterator,bool> ret = aggrBuffer.insert(std::make_pair(pseudoLeaf,reinterpret_cast<long int>(bufferStack)));
		assert(ret.second);
		bufIter=ret.first;
	}

	if(aggrSize >= aggrBlockSize)
	{
		wlManager->SetAsCurrentBlockStack2(bufferStack);
		msgTraverse.l.clear();
		msgTraverseRight.l.clear();
		blockId = wlManager->GetBufferData(vis, msgTraverse.l, msgTraverseRight.l);
		msgTraverse.blkStart = blockId;
		msgTraverseRight.blkStart = blockId;
		aggrBuffer.erase(bufIter);
	}

	return aggrSize;
}


/* Description: This function is called to send compressed messagess.
 * compressed message buffers may exist at multiple pseudo leaves (At different heights) within the same process. Some may have been processed and some may not.
 * Hence it is necessary to go through entire aggrBuffer to and then send all the buffers that have not been processed.
 * Parameters: None
 * Return Value: None
 */
void SPIRIT::SPIRIT_TraverseHelper_SendCompressedMessages()
{
	int childDesc;
	long int child;
	//search for the entry in the buffer for this 'node'
	std::map<long int, long int>::iterator bufIter = aggrBuffer.begin();
	while(bufIter != aggrBuffer.end())
	{
		MsgTraverse msgTraverse, msgTraverseRight;
		msgTraverse.type  = curTreeType;
		msgTraverseRight.type  = curTreeType;
		TBlockId blockId;
		Vertex* pLeaf = reinterpret_cast<Vertex*>(bufIter->first);
		wlManager->SetAsCurrentBlockStack2(reinterpret_cast<BlockStack*>(bufIter->second));
		blockId = wlManager->GetBufferData(vis, msgTraverse.l, msgTraverseRight.l);
		if(msgTraverse.l.size() > 0)
		{
			msgTraverse.blkStart = blockId;
			msgTraverse.pLeaf = bufIter->first;
			if(curTreeType == OCTREE)
			{
				for(int i=0;i<8;i++)
				{
					if(pLeaf->pChild[i])
					{
						child = reinterpret_cast<long int>(pLeaf->pChild[i]);
						pLeaf->GetPLeafChildrenDetails(pLeaf->pChild[i], msgTraverse);
						childDesc = pLeaf->childDesc[i];
						break;
					}
				}
				msgTraverse.pRoot = child;
			}
			else
				msgTraverse.pRoot = reinterpret_cast<long int>(pLeaf->leftChild);
			//msgTraverse.pLeafThreshold = pLeaf->threshold;
			if(pLeaf->rightChild)
			{
				msgTraverse.pSibling = reinterpret_cast<long int>(pLeaf->rightChild);
				msgTraverse.siblingDesc = pLeaf->rightDesc;
			}
			else
			{
				msgTraverse.pSibling = static_cast<long int>(0);
			}
			
			if(curTreeType == OCTREE)
				SendMessage(childDesc,MESSAGE_TRAVERSE, &msgTraverse);
			else
				SendMessage((pLeaf->leftDesc),MESSAGE_TRAVERSE, &msgTraverse);
		}

		if(msgTraverseRight.l.size() > 0)
		{
			msgTraverseRight.blkStart = blockId;
			msgTraverseRight.pRoot = reinterpret_cast<long int>(pLeaf->rightChild);
			msgTraverseRight.pLeaf = bufIter->first;
			if(pLeaf->leftChild)
			{
				msgTraverseRight.pSibling = reinterpret_cast<long int>(pLeaf->leftChild);
				msgTraverseRight.siblingDesc = pLeaf->leftDesc;
			}
			else
			{
				msgTraverseRight.pSibling = static_cast<long int>(0);
			}
			SendMessage((pLeaf->rightDesc),MESSAGE_TRAVERSE, &msgTraverseRight);
		}
#ifdef STATISTICS		
				bufferStage += msgTraverse.l.size() + msgTraverseRight.l.size();
#endif

		aggrBuffer.erase(bufIter);
		bufIter++;
	}

	return;	
}

bool SPIRIT::SPIRITHelper_HandleMessageTraverseBackward_Multiple(Vertex* parentNode, MsgTraverse& msg, int msgId)
{
	bool decodeBlocks = false;
	TCBSet blkIdSet;
	int status = wlManager->GetCompressedBlockIDs(msg.blkStart,msg.l[0],blkIdSet);
	if(status == STATUS_SUCCESS)
	{
		if(msgId == MESSAGE_TRAVERSE_BACKWARD)
			wlManager->SetContext(vis,msg.l, msg.blkStart, false);
			
		decodeBlocks = true;
		int ret = 1;
		TCBSet::iterator sIter = blkIdSet.begin();

		for(;sIter!=blkIdSet.end();sIter++)
		{
			wlManager->SetAsCurrentBlockStack2(*sIter);
			TBlockId curBlockId = (*sIter)->bStackId;
			status = SPIRITHelper_HandleMessageTraverseBackward(parentNode,curBlockId);
			if(status != 1)
			{
				ret = STATUS_TRAVERSE_INCOMPLETE;
				continue;
			}
			else
			{
				int re = wlManager->RemoveBlockStack(vis);		
			}
		}
		
	}
	return decodeBlocks;
	
}


#endif

int SPIRIT::SPIRITHelper_DeAggregateBlockAndSend(Vertex** pRoot, MsgTraverse& msgTraverse, bool loadBalanced)
{
	int ret = STATUS_TRAVERSE_COMPLETE;
	bool traverseIncomplete = false;
	bool siblingTraversalPending = false;
	int startIndx = 0, endIndx = 0;
	
	Vertex* tmpVertex = *pRoot;
	SPIRIT_PseudoRoot* pSibling=NULL;
 	int siblingDesc;
	
	if(curTreeType != OCTREE)
	{
		pSibling = reinterpret_cast<SPIRIT_PseudoRoot*>(msgTraverse.pSibling);
		if(pSibling != 0)
			siblingDesc = msgTraverse.siblingDesc;
	}
	else
	{
		if(msgTraverse.pSiblings.size() > 0)
		{
			pSibling = reinterpret_cast<SPIRIT_PseudoRoot*>(msgTraverse.pSiblings[0]);
			siblingDesc = msgTraverse.siblingDescs[0];
			msgTraverse.pSiblings.erase(msgTraverse.pSiblings.begin());
			msgTraverse.siblingDescs.erase(msgTraverse.siblingDescs.begin());
		}
	}
	BlockStackListElem curBStack = wlManager->GetCurrentBlockStack();
	

#ifdef MESSAGE_AGGREGATION
	std::list<BlockStack*> tmpFTable;
	Vertex* nextNodeToVisit=NULL;

	if(msgTraverse.l.size() > opts.blockSize)
	{
		int totalBlocks = msgTraverse.l.size()/opts.blockSize;
		if(msgTraverse.l.size()%opts.blockSize)
			totalBlocks+=1;
		while(totalBlocks)
		{
			endIndx = startIndx + opts.blockSize;
			if(endIndx > msgTraverse.l.size())
				endIndx = msgTraverse.l.size();
			TIndices workBlock(endIndx-startIndx);
			int j=0;
			for(int i=startIndx;i<endIndx;i++,j++)
				workBlock[j]=(*(msgTraverse.l.begin()+i))->index;
			
			BlockStack* fragBStack = wlManager->CreateBlockStack(workBlock, tmpVertex, NULL, 2, curBStack, procRank);
			busyTime.Start();
			ret = SPIRIT_TraverseHelper(tmpVertex, NULL);
			busyTime.Stop();
			
			if(ret == STATUS_TRAVERSE_INCOMPLETE)
			{
				traverseIncomplete = true;
			}
			else if(ret == STATUS_TRAVERSE_COMPLETE)
			{
				if(!pSibling || (pSibling && (siblingDesc != procRank)) || (traverseIncomplete) )
				{
					int nextNodeProc=0;
					wlManager->PopFromCurrentBlockStackAndUpdate();
					wlManager->RemoveBlockStack(vis);
				}
				else
				{
					tmpFTable.push_front(fragBStack);
				}
			}
					
			startIndx += opts.blockSize;
			totalBlocks--;
		}

		if(traverseIncomplete)
		{
			std::list<BlockStack*>::iterator fIter = tmpFTable.begin();
			for(;fIter != (tmpFTable).end();fIter++)
			{
				wlManager->SetAsCurrentBlockStack2(*fIter);
				wlManager->PopFromCurrentBlockStackAndUpdate();
				wlManager->RemoveBlockStack(vis);
			}
		}

		while(!traverseIncomplete)
		{	
			if(pSibling)
			{
				if(siblingDesc == procRank)	
				{
					tmpVertex = pSibling;
					if(curTreeType != OCTREE)
						pSibling = NULL;
					else
					{
						pSibling = NULL;
						if(msgTraverse.pSiblings.size() > 0)
						{
							pSibling = reinterpret_cast<SPIRIT_PseudoRoot*>(msgTraverse.pSiblings[0]);
							siblingDesc = msgTraverse.siblingDescs[0];
							msgTraverse.pSiblings.erase(msgTraverse.pSiblings.begin());
							msgTraverse.siblingDescs.erase(msgTraverse.siblingDescs.begin());
						}
					}
					int nextNodeProc = 0;
					std::list<BlockStack*>::iterator fIter = tmpFTable.begin();
					for(;fIter != (tmpFTable).end();fIter++)
					{
						wlManager->SetAsCurrentBlockStack2(*fIter);
						busyTime.Start();
						ret = SPIRIT_TraverseHelper(tmpVertex, NULL);
						busyTime.Stop();
						if(ret == STATUS_TRAVERSE_COMPLETE)
						{
							if(!pSibling)
							{
								wlManager->PopFromCurrentBlockStackAndUpdate();
								wlManager->RemoveBlockStack(vis);
							}
						}
						else if(ret == STATUS_TRAVERSE_INCOMPLETE)
						{
							traverseIncomplete = true;
						}
					}
					/*if(!traverseIncomplete)
					{
						if(curTreeType != OCTREE)
							pSibling = NULL;
						else
						{
							pSibling = NULL;
							if(msgTraverse.pSiblings.size() > 0)
							{
								pSibling = reinterpret_cast<SPIRIT_PseudoRoot*>(msgTraverse.pSiblings[0]);
								siblingDesc = msgTraverse.siblingDescs[0];
								msgTraverse.pSiblings.erase(msgTraverse.pSiblings.begin());
								msgTraverse.siblingDescs.erase(msgTraverse.siblingDescs.begin());
							}
						}
					}*/
				}
				else
				{
					siblingTraversalPending = true;
					break;
				}
			}
			else
				break;
		}

		if(siblingTraversalPending)
		{
			int nextNodeProc=0;
			assert(tmpVertex == *pRoot);
			MsgTraverse tmp;
			tmp.type  = curTreeType;
			tmp.pLeaf = reinterpret_cast<long int>(((Vertex*)(*pRoot))->parent);
			tmp.pRoot = reinterpret_cast<long int>(pSibling);
			SPIRIT_PseudoRoot* psRoot = (SPIRIT_PseudoRoot *)(*pRoot);
			tmp.pSibling = 0;//reinterpret_cast<long int>(psRoot->pSibling);
			tmp.pSiblings = msgTraverse.pSiblings;
			tmp.siblingDescs = msgTraverse.siblingDescs;
			tmp.blkStart = msgTraverse.blkStart;
			wlManager->SetAsCurrentBlockStack2(curBStack);
			wlManager->GetContext(vis,tmp.l); //msgTraverse.blkStart is different from local blkStart returned by GetContext
			wlManager->RemoveBlockStack(vis);	
			SendMessage(siblingDesc,MESSAGE_TRAVERSE, &tmp);
			traverseIncomplete = true;
		}

		if(traverseIncomplete)
		{
			ret = STATUS_TRAVERSE_INCOMPLETE;
		}
		else
		{
			int nextNodeProc=0;
			wlManager->DeleteFromSuperBlock(tmpVertex, &nextNodeToVisit, nextNodeProc);
			assert(ret == STATUS_TRAVERSE_COMPLETE);
			wlManager->SetAsCurrentBlockStack2(curBStack);
			wlManager->FreeContext(msgTraverse.l);
			msgTraverse.l.clear();
			wlManager->GetContext(vis,msgTraverse.l);
			*pRoot = tmpVertex;
		}
	}
	else
#endif
	do
	{
		busyTime.Start();
		ret = SPIRIT_TraverseHelper(*pRoot, pSibling);
		busyTime.Stop();
		if(ret == STATUS_TRAVERSE_COMPLETE)
		{
			if(pSibling)
			{
				Vertex* nextNodeToVisit=NULL;
				int nextNodeProc;
				TBlockId blkStart = wlManager->DeleteFromSuperBlock(*pRoot, &nextNodeToVisit, nextNodeProc);
				wlManager->FreeContext(msgTraverse.l);
				msgTraverse.l.clear();
				wlManager->GetContext(vis,msgTraverse.l);
				if(siblingDesc != procRank)
				{
					//pLeafThreshold nd parentCoords remain the same.
					//sending entire compressed data to sibling
					msgTraverse.pRoot = reinterpret_cast<long int>(pSibling);
					msgTraverse.pSibling = static_cast<long int>(0);
					msgTraverse.blkStart = blkStart;
					if(!loadBalanced)
					{
						int re = wlManager->RemoveBlockStack(vis);	
							if(re == -1)
								printf("Debug 1\n");
					}
					SendMessage((msgTraverse.siblingDesc),MESSAGE_TRAVERSE, &msgTraverse);
					ret = STATUS_TRAVERSE_INCOMPLETE;
				}
				else
				{
					*pRoot = pSibling;
					pSibling = NULL;
					if(curTreeType == OCTREE)
					{
						if(msgTraverse.pSiblings.size() > 0)
						{
							pSibling = reinterpret_cast<SPIRIT_PseudoRoot*>(msgTraverse.pSiblings[0]);
							siblingDesc = msgTraverse.siblingDescs[0];
							msgTraverse.pSiblings.erase(msgTraverse.pSiblings.begin());
							msgTraverse.siblingDescs.erase(msgTraverse.siblingDescs.begin());
						}
					}
					if((*pRoot)->level==opts.subtreeHeight)
						((SPIRIT_PseudoRoot*)(*pRoot))->parents2[msgTraverse.blkStart.first] = msgTraverse.pLeaf;
					continue;
					//ret = SPIRIT_TraverseHelper(*pRoot,pSibling);
				}
			}
	
			if(ret == STATUS_TRAVERSE_COMPLETE)
			{
				Vertex* nextNodeToVisit=NULL;
				int nextNodeProc=0;
				TBlockId blkStart = wlManager->DeleteFromSuperBlock(*pRoot, &nextNodeToVisit, nextNodeProc);	
				wlManager->FreeContext(msgTraverse.l);
				msgTraverse.l.clear();
				wlManager->GetContext(vis,msgTraverse.l);
				break;
			}

		}
	}while(ret == STATUS_TRAVERSE_COMPLETE);
	
	return ret;
}


int SPIRIT::SPIRITHelper_UpdatePLeaves_Oct(MsgUpdatePLeaves_Oct& msg)
{
	int ret = STATUS_SUCCESS;
	std::vector<MsgUpdatePLeaf_Oct>::iterator msgIter = msg.vPLeaves.begin();

	for(msgIter = msg.vPLeaves.begin();msgIter != msg.vPLeaves.end();msgIter++)
	{
		Vertex* pLeaf = NULL;
		bool found=false;
		std::map<long int, PLeafPointBucket>::iterator pLeafIter;
		for(pLeafIter = pLeafMap.begin();pLeafIter!=pLeafMap.end();pLeafIter++)
		{
			pLeaf = reinterpret_cast<Vertex*>(pLeafIter->first);
			if(pLeaf->label == msgIter->label)
			{
				pLeafIter->second.numVerticesInSubtree = msgIter->numVertices;
				found = true;
				break;
			}
		}	
		assert(found);
		if(curTreeType == OCTREE)
		{
			OctreeVertexData* vData  = (OctreeVertexData*)(pLeaf->vData);
			vData->cofm = msgIter->cofm;
			vData->mass = msgIter->mass;
		}

		for(int i=0;i<msgIter->cell.size();i++)
		{
			char childNum = msgIter->cell[i];
			if(curTreeType == OCTREE)
			{
				pLeaf->pChild[childNum] = reinterpret_cast<Vertex*>(msgIter->children[i]);
				pLeaf->childDesc[childNum] = msgIter->descs[0];
			}
			else
			{
				if(childNum == 0)
				{
					pLeaf->leftChild = reinterpret_cast<Vertex*>(msgIter->children[i]);
					pLeaf->leftDesc = msgIter->descs[0];
					
				}
				else
				{
					pLeaf->rightChild = reinterpret_cast<Vertex*>(msgIter->children[i]);
					pLeaf->rightDesc = msgIter->descs[0];
				}
			}

			if(curTreeType == OCTREE)
			{
				if(msgIter->descs[0] == procRank)
				{
					SPIRIT_PseudoRoot* p = (SPIRIT_PseudoRoot*)(pLeaf->pChild[childNum]);
					p->parents2.insert(std::make_pair(procRank,reinterpret_cast<long int>(pLeaf)));
				}
			}
		}
	}
	return ret;
}


int SPIRIT::SPIRITHelper_GetNextProcessId(long int pLeaf)
{
	//get next process and send the message
	/*int nextprocessid = ((procRank)* 2 + 1 + procCount) % numProcs;
	procCount++;
	if(procCount > (1<<(opts.subtreeHeight)))
	procCount = 1;
	if(nextprocessid == procRank)
	{
	nextprocessid +=1;
	if(nextprocessid == numProcs)
		nextprocessid = 0;
	}*/
	int nextprocessid;
	{
		PLeafPointBucket tmp;
		std::pair<std::map<long int, PLeafPointBucket>::iterator, bool> ret = pLeafMap.insert(std::make_pair(pLeaf,tmp));
		if(ret.second == false)
		{
			if((ret.first)->first !=  pLeaf)
				printf("Hash table conflict\n");
			nextprocessid = (ret.first)->second.subtreeOwner;
		}
		else
		{
			/*do
			{
				nextprocessid = ((procRank)* 2 + 1 + procCount) % numProcs;
				procCount++;
				if(procCount > (1<<(opts.subtreeHeight * 3)))
					procCount = 1;
			}while(nextprocessid == procRank);*/
			nextprocessid = procCount % numProcs;
			procCount++;

			(ret.first)->second.subtreeOwner = nextprocessid;
			Vertex* pseudoLeaf = reinterpret_cast<Vertex*>(pLeaf);
			//(ret.first)->second.label = pseudoLeaf->label;
			(ret.first)->second.numVerticesInSubtree = 0;
		}
	}
	return nextprocessid;
}


void SPIRIT::SPIRITHelper_CountSubtreeNodes(Vertex* ver, long int& count)
{
		int ret = STATUS_TRAVERSE_COMPLETE;
#ifdef TRAVERSAL_PROFILE
		count += ver->pointsVisited;
#else
		count++;
#endif
		assert(ver != NULL);

		if((ver->leftChild == NULL) && (ver->rightChild==NULL))
		{
			return;
		}
		if(ver->level == opts.subtreeHeight-1)
			return;
		
		if(ver->leftChild)
		{
			if(ver->leftDesc == procRank)
				SPIRITHelper_CountSubtreeNodes(ver->leftChild,count);
		}
		if(ver->rightChild)
		{
			if(ver->rightDesc == procRank)
				SPIRITHelper_CountSubtreeNodes(ver->rightChild,count);
		}
}

void SPIRIT::SPIRITHelper_CountSubtreeNodes_Oct(Vertex* ver, long int& count)
{
		int ret = STATUS_TRAVERSE_COMPLETE;
		count++;
		assert(ver != NULL);

		if(ver->leaf)
		{
			return;
		}

		if(ver->level == opts.subtreeHeight-1)
			return;
		
		for(int i=0;i<ver->pChild.size();i++)
		{
			if(ver->childDesc[i] == procRank)
			{
				SPIRITHelper_CountSubtreeNodes_Oct(ver->pChild[i],count);
			}
		}
}

#ifdef TRAVERSAL_PROFILE
void SPIRIT::SPIRITHelper_GetTotalPointsVisited(Vertex* ver, long int& count)
{
		int ret = STATUS_TRAVERSE_COMPLETE;
		count += ver->pointsVisited;
		assert(ver != NULL);

		if(curTreeType == OCTREE)
		{
			if(ver->leaf)
			{
				return;
			}
			
			for(int i=0;i<ver->pChild.size();i++)
			{
				if(ver->childDesc[i] == procRank)
				{
					SPIRITHelper_GetTotalPointsVisited(ver->pChild[i],count);
				}
			}
		}
		else
		{
			if((ver->leftChild == NULL) && (ver->rightChild==NULL))
			{
				return;
			}
			
			if(ver->leftChild)
			{
				if(ver->leftDesc == procRank)
					SPIRITHelper_GetTotalPointsVisited(ver->leftChild,count);
			}
			if(ver->rightChild)
			{
				if(ver->rightDesc == procRank)
					SPIRITHelper_GetTotalPointsVisited(ver->rightChild,count);
			}
		}
}
#endif

void SPIRIT::print_treetofile(FILE* fp)
{
	print_preorder(rootNode, fp);
}

void SPIRIT::print_preorder(Vertex* node, FILE* fp)
{
	VptreeVertexData* vData = (VptreeVertexData*)(node->vData);
	fprintf(fp,"%d %d %f\n",node->label, node->level, vData->threshold);
	/*for (int j = 0; j < DIMENSION; j++) 
	{
		fprintf(fp,"%f %f ",node->max_d[j],node->min_d[j]);
	}
	fprintf(fp,"\n");*/
	if(node->leftChild)
		print_preorder(node->leftChild,fp);

	if(node->rightChild)
		print_preorder(node->rightChild,fp);
}

long int SPIRIT::ReadTreeData_Oct(char* inputFile, long int totalPointsRead, long int numPoints, unsigned char* pointArr, InputFileParser* inputParser, int step)
{
	long int startIndex=0, endIndex=numPoints, tmpIndex=0;
	long int numRecordsToSkip=0;
	
	if((inputParser->GetTreeType() == KDTREE) ||(inputParser->GetTreeType() == VPTREE))
		numRecordsToSkip = numPoints/2;

	if(step==0)
	{
		input.open(inputFile, std::fstream::in);
		if(input.fail())
		{
			std::cout<<"File does not exist. exiting"<<std::endl;
			return 0;
		}
		if(numPoints > opts.batchSize)
		{
			startIndex = procRank * numPoints/numProcs;
			endIndex = (procRank == numProcs-1)?(numPoints):((procRank+1) * numPoints/numProcs);
		}
		/*else
		{
			if(procRank > 9)
				return 0;
		}*/
		numRecordsToSkip = startIndex+5;
		endIndex += 5;
	}
	else if(step==1)
	{
		input.seekg(0, input.beg);
		if(curTreeType == OCTREE)
		{
			numRecordsToSkip = 5;
			endIndex += 5;
		}
	}
	int offsetFactor=0;
	if(curTreeType == OCTREE)
		offsetFactor = sizeof(OctreePoint);
	else if(curTreeType == KDTREE)
		offsetFactor = sizeof(KdtreePoint);
	else if(curTreeType == PCKDTREE)
		offsetFactor = sizeof(PCKdtreePoint);
	else if(curTreeType == VPTREE)
		offsetFactor = sizeof(VptreePoint);

        while(true) 
	{
	    Point* p = 	(Point *) (pointArr+(tmpIndex*offsetFactor));

	    if(curTreeType == OCTREE)
	    {
		if(totalPointsRead < 5)
			p  = NULL;
	    }

	    parser->ReadPoint(input, p);
	    totalPointsRead++;
	    if(totalPointsRead <= numRecordsToSkip)
	    {
		continue;
	    }
	    tmpIndex++;
	    p->id = totalPointsRead;
	    if(((numProcs>1) && tmpIndex==opts.batchSize)||(totalPointsRead == endIndex))
		break;
        }
	
	return tmpIndex;
}


void SPIRIT::ReadTreeData(char* inputFile, long int numPoints, std::vector<Point*>& points, InputFileParser* parser)
{
	input.open(inputFile, std::fstream::in);
	if(input.fail())
	{
		std::cout<<"File does not exist. exiting"<<std::endl;
		return;
	}
	curTreeType = parser->GetTreeType();
	int offsetFactor=0;
	if(curTreeType == OCTREE)
		offsetFactor = sizeof(OctreePoint);
	else if(curTreeType == KDTREE)
		offsetFactor = sizeof(KdtreePoint);
	else if(curTreeType == PCKDTREE)
		offsetFactor = sizeof(PCKdtreePoint);
	else if(curTreeType == VPTREE)
		offsetFactor = sizeof(VptreePoint);

	long int startIndex, endIndex;
	long int totalPointsRead = 0, numRecordsToSkip=0;
	startIndex = 0;
	endIndex = numPoints;
	if(curTreeType == OCTREE)
	{
		numRecordsToSkip = 5;
		endIndex += 5;
	}

        while(true) 
	{
	    totalPointsRead++;
	
	    Point* p = NULL;
	    if(totalPointsRead > numRecordsToSkip)
	    {
		if(curTreeType == OCTREE)
			p = new OctreePoint();
		else if(curTreeType == KDTREE)
			p = new KdtreePoint();
		else if(curTreeType == PCKDTREE)
			p = new PCKdtreePoint;
		else if(curTreeType == VPTREE)
			p = new VptreePoint();
	    }
	    parser->ReadPoint(input, p);
	    if(!p)
	    {
		continue;
	    }
	    p->id = totalPointsRead;
	    points.push_back(p);
	    if(totalPointsRead == endIndex)
	    {
		break;
	    }
        }
	return;
}

void SPIRIT::AggregateResults()
{
	long int nodesTraversed_iteration=0, totalNodesTraversed_iteration;
	long int corrsum_iteration=0, totalcorrsum_iteration;
	int offsetFactor=0;
	if(curTreeType == OCTREE)
		offsetFactor = sizeof(OctreePoint);
	else if(curTreeType == KDTREE)
		offsetFactor = sizeof(KdtreePoint);
	else if(curTreeType == PCKDTREE)
		offsetFactor = sizeof(PCKdtreePoint);
	else if(curTreeType == VPTREE)
		offsetFactor = sizeof(VptreePoint);
			
#if 0
	for(long int i=wlManager->start;i<wlManager->end;i++)
	{
		Point* p = (Point*)(wlManager->points+i*offsetFactor);
		nodesTraversed_iteration += p->nodesTraversed;
		/*if(curTreeType == KDTREE)
		{
			KdtreePoint* kdp = (KdtreePoint*)(p);
			printf("%d (%f %f %f %f%f %f %f) %f %ld\n",kdp->id, kdp->pt[0],kdp->pt[1], kdp->pt[2],kdp->pt[3],kdp->pt[4],kdp->pt[5], kdp->pt[6],kdp->closest_dist, kdp->closest_label);
		}
		if(curTreeType == VPTREE)
		{
			VptreePoint* vpp = (VptreePoint*)(p);
			printf("%d (%f %f %f %f%f %f %f) %f %ld\n",vpp->id, vpp->pt[0],vpp->pt[1], vpp->pt[2],vpp->pt[3],vpp->pt[4],vpp->pt[5], vpp->pt[6],vpp->tau, vpp->closest_label);
		}*/
		if(curTreeType == PCKDTREE)
		{
			PCKdtreePoint* kdp = (PCKdtreePoint*)(p);
			corrsum_iteration += kdp->corr;
			//printf("%d (%f %f %f %f%f %f %f) %ld %ld\n",kdp->id, kdp->pt[0],kdp->pt[1], kdp->pt[2],kdp->pt[3],kdp->pt[4],kdp->pt[5], kdp->pt[6],kdp->corr, kdp->nodesTraversed);
			//printf("%d (%f %f) %ld\n",kdp->id, kdp->pt[0],kdp->pt[1],kdp->corr);
		}
		/*if(curTreeType == OCTREE)
		{
			OctreePoint* op = reinterpret_cast<OctreePoint*>(p);
			printf("%d (%f %f %f) %ld\n",op->id,op->cofm.pt[0],op->cofm.pt[1], op->cofm.pt[2], op->nodesTraversed);
		}*/
	}

	if(curTreeType == PCKDTREE)
	{
		reduce(communicator(pg),corrsum_iteration, totalcorrsum_iteration, std::plus<long int>(),0);
		if(procRank == 0)
			printf("KdTree correlation: %f\n",(((float)totalcorrsum_iteration)/wlManager->numPoints));
	}
#endif
	
	reduce(communicator(pg),wlManager->totalNodesTraversed, totalNodesTraversed_iteration, std::plus<long int>(),0);
	if(procRank == 0)
		totalNodesTraversed = totalNodesTraversed_iteration;
}


void SPIRIT::PrintResults()
{
	if(procRank == 0)
		printf("total Nodes traversed %ld\n",totalNodesTraversed);
}

SPIRIT* SPIRIT::GetInstance(Optimizations& opts, mpi_process_group& prg)
{
		if(instance == NULL)
			instance  = new SPIRIT(opts, prg);

		return instance;
}

float SPIRIT::ConstructOctree(char* fileName, long int numPoints, InputFileParser* inputParser)
{
	PBGL_oobMessage msg;
	bool remoteNodePresent = false, done =false;
	Vec center;
	float dia;
	int numAcksReceived = 1;
	SPIRIT_PseudoRoot* curSubtreeBuilt=NULL;
	int expectedUpdates=0;
	MsgUpdatePLeaves_Oct msgUpdatePLeaves;
	double startTime, endTime, constnTime=0.;
	int offsetFactor = sizeof(OctreePoint);
	

	numTraversals = numPoints;
	parser=inputParser;
	curTreeType = parser->GetTreeType();

	unsigned char* points = NULL;
	/*if(numPoints > opts.batchSize)
		address = new unsigned char[opts.batchSize*sizeof(OctreePoint)];	
	else
		address = new unsigned char[numPoints*sizeof(OctreePoint)];*/
	AllocateWorkspace(numPoints, inputParser);
	points = address;
	
	long int totalPointsRead = 0;
	int step=0;
	while(totalPointsRead < numPoints)
	{
		synchronize(pg);
		int numRead = ReadTreeData_Oct(fileName, totalPointsRead, numPoints, points, inputParser, step);
		if(numRead == 0)
		{
			printf("ERROR. Tree Construction failed\n");
			return 0;
		}
		startTime=clock();
		step++;
		if(step == 1)
		{
			SPIRITHelper_ComputeBoundingBoxParams(points, numRead, numPoints, center, dia);
			if(!rootNode)
			{
				rootNode = CreateVertex(VTYPE_NORMAL);
				rootNode->vData = new OctreeVertexData(NULL);
				rootNode->level = 0;
			}
			if(numPoints > opts.batchSize)
				continue;
		}
		totalPointsRead += numRead;
		//synchronize(pg);
		for (int i=0; i<numRead; i++)
		{
			OctreePoint* p = reinterpret_cast<OctreePoint*>(points+i*offsetFactor);
			int doneBuildSubTree = BuildSubTree_Oct(rootNode, procRank, p,center, (dia*0.5),1,1, true);
			if(doneBuildSubTree == BUILDSUBTREE_SAMEPROCESS)
			{
				if(i != numRead-1 )
					continue;
			}
			else 
				remoteNodePresent = true;
		}
		endTime = clock();
		constnTime += (endTime-startTime)/CLOCKS_PER_SEC;
	}

	startTime = clock();
	if(!remoteNodePresent)
		done = true;
	
	
	/*{
		printf("Rank0 IDs\n");
		for(int i=0;i<rank0PseudoLeaves.size();i++)
		{
			Vertex* pLeaf = rank0PseudoLeaves[i];
			printf("%ld ",pLeaf->label);
		}
		printf("\n");
		
	}*/

	for(int i=0;i<rank0PseudoLeaves.size();i++)
	{
		Vertex* pLeaf = rank0PseudoLeaves[i];
		/*this is done to let the recursion in SPIRITHelper_ComputeCofm pass through the pseudoleaf. 
		Currently, SPIRITHelper_ComputCofm stops computing cofm at the level of pseudoleaves since the cofm of all vertices in subtrees below are already computed by other ranks */
		bool explorePseudoLeafChildren=true;
		SPIRITHelper_ComputeCofm(pLeaf, explorePseudoLeafChildren);
		MsgUpdatePLeaf_Oct m;
		m.descs.push_back(-1);
		for(int i=0;i<pLeaf->pChild.size();i++)
		{
			if(pLeaf->pChild[i])
			{
				m.descs[0]=pLeaf->childDesc[i];
				break;
			}
		}
		for(int i=0;i<pLeaf->pChild.size();i++)
		{
			if(pLeaf->pChild[i])
			{
				m.cell.push_back(i);
				m.children.push_back(reinterpret_cast<long int>(pLeaf->pChild[i]));
			}
		}
		OctreeVertexData* vData = ((OctreeVertexData*)(pLeaf->vData));
		m.label = pLeaf->label;
		m.cofm = vData->cofm;
		m.mass = vData->mass;
		m.numVertices=0;
		pLeaf->level++; 
		SPIRITHelper_CountSubtreeNodes_Oct(pLeaf, m.numVertices);
		pLeafMap[reinterpret_cast<long int>(pLeaf)].numVerticesInSubtree = m.numVertices;
		pLeaf->level--; 
		msgUpdatePLeaves.vPLeaves.push_back(m);
	}
	
	if(procRank == 0)
	{
		std::map<int, int> uniqBottomLevelNodes;
		std::map<long int,PLeafPointBucket>::iterator iter=pLeafMap.begin();
		for(;iter!=pLeafMap.end();iter++)
		{
			std::pair<std::map<int, int>::iterator,bool> isUniq = uniqBottomLevelNodes.insert(std::make_pair(iter->second.subtreeOwner, 1));
			if(isUniq.second)
				expectedUpdates++;
		}
		expectedUpdates--;
	}
	else
		expectedUpdates = 1;

	if(msgUpdatePLeaves.vPLeaves.size() > 0)
	{
		if(procRank !=0)
		{
			SendMessage(0,MESSAGE_UPDATE_PLEAVES_OCT,&msgUpdatePLeaves); 
		}
		msgUpdatePLeaves.vPLeaves.clear();
	}

	while(!done)
	{
		//poll for messages
		PBGL_AsyncMsg pollMsg = pg.poll();
		if(!pollMsg)
			continue;
		else
			msg = pollMsg.get();
		switch(msg.second)
		{
				case MESSAGE_UPDATE_PLEAVES_OCT:
				{
					MsgUpdatePLeaves_Oct msgUpdatePLeaves;
					receive_oob(pg, msg.first, msg.second, msgUpdatePLeaves);
					int status = SPIRITHelper_UpdatePLeaves_Oct(msgUpdatePLeaves);
					expectedUpdates--;
					if(expectedUpdates == 0)
					{
						if(procRank == 0)
						{
							MsgUpdatePLeaves_Oct msgUpdatePLeaves;
							std::map<long int,PLeafPointBucket>::iterator iter=pLeafMap.begin();
							for(;iter!=pLeafMap.end();iter++)
							{
								Vertex* pLeaf = reinterpret_cast<Vertex*>(iter->first);
								MsgUpdatePLeaf_Oct m;
								m.descs.push_back(-1);
								for(int i=0;i<pLeaf->pChild.size();i++)
								{
									if(pLeaf->pChild[i])
									{
										m.descs[0]=pLeaf->childDesc[i];
										break;
									}
								}
								for(int i=0;i<pLeaf->pChild.size();i++)
								{
									if(pLeaf->pChild[i])
									{
										m.cell.push_back(i);
										m.children.push_back(reinterpret_cast<long int>(pLeaf->pChild[i]));
										//m.descs.push_back(pLeaf->childDesc[i]);
									}
								}
								OctreeVertexData* vData = ((OctreeVertexData*)(pLeaf->vData));
								m.label = pLeaf->label;
								m.cofm = vData->cofm;
								m.mass = vData->mass;
								msgUpdatePLeaves.vPLeaves.push_back(m);
							}

							for(int i=1;i<numProcs;i++)
							{
								SendMessage(i,MESSAGE_UPDATE_PLEAVES_OCT,&msgUpdatePLeaves); 
							}
						}	
						done = true;
					}
					/*if(!msgUpdatePLeaves.moreData)
						done = true;
					else
					{
						int lastSent = msgUpdatePLeaves.lastSent;
						SendMessage(msg.first,MESSAGE_UPDATE_PLEAVES_ACK,&lastSent); 
					}*/
				}
				break;
				/*case MESSAGE_UPDATE_PLEAVES_ACK:
				{
					int lastSent;
					receive_oob(pg, msg.first, msg.second, lastSent);
					numAcksReceived++;
					if(numAcksReceived < numProcs)
					{
						break;
					}
					numAcksReceived =1;
					MsgUpdatePLeaves_Oct msgUpdatePLeaves;
					std::vector<MsgUpdatePLeaf_Oct*>::iterator iter = pLeafBucket2.begin()+lastSent+1;
					int i=0;
					for(;iter!=pLeafBucket2.end();iter++)
					{
						i++;
						msgUpdatePLeaves.vPLeaves.push_back(*(*iter));
						delete *iter;
						if(i == 5000)
						{
							msgUpdatePLeaves.moreData = true;
							break;
						}
					}
					msgUpdatePLeaves.lastSent = lastSent+i;
					//pLeafBucket2.erase(pLeafBucket2.begin(),pLeafBucket2.begin()+i);
					for(int i=1;i<numProcs;i++)
					{
						SendMessage(i,MESSAGE_UPDATE_PLEAVES_OCT,&msgUpdatePLeaves); 
					}
					if(!msgUpdatePLeaves.moreData)
						done = true;
				}
				break;*/
			default:break;
		}
		
	}
	endTime = clock();
	constnTime += (endTime-startTime)/CLOCKS_PER_SEC;
	if(procRank == 0)
	{
		printf("%d number of pseudoleaves %d\n",procRank,pLeafMap.size());
		printf("Construction Done. Realigning and Computing Cofm\n");
	}
	/*if(procRank == 0)
	{
	printf("PLeafMap IDs\n");
	std::map<long int,PLeafPointBucket>::iterator iter=pLeafMap.begin();
	for(;iter!=pLeafMap.end();iter++)
	{
		Vertex* pLeaf = reinterpret_cast<Vertex*>(iter->first);
		if(pLeaf->pseudoLeaf)
		{
			printf("%ld (%d-%p %d-%p %d-%p %d-%p %d-%p %d-%p %d-%p %d-%p)",pLeaf->label,pLeaf->childDesc[0],pLeaf->pChild[0],pLeaf->childDesc[1],pLeaf->pChild[1],pLeaf->childDesc[2],pLeaf->pChild[2],pLeaf->childDesc[3],pLeaf->pChild[3],pLeaf->childDesc[4],pLeaf->pChild[4],pLeaf->childDesc[5],pLeaf->pChild[5],pLeaf->childDesc[6],pLeaf->pChild[6],pLeaf->childDesc[7],pLeaf->pChild[7]);
			bool flag=false;
			for(int i=0;i<8;i++)
			{
				if((pLeaf->pChild[i]) && (pLeaf->childDesc[i]==procRank) && pLeaf->pChild[i]->leaf)
				{
					flag=true;
					break;
				}
			}
			if(flag)
				printf(" true\n");
			else
				printf(" false\n");
		}
	}
	printf("\n");
	}*/
	startTime=clock();
	SPIRITHelper_ComputeCofm(rootNode);	
	/*OctreeVertexData* vD=(OctreeVertexData*)(rootNode->vData);
	printf("root cofm %f %f %f\n",vD->cofm.pt[0],vD->cofm.pt[1],vD->cofm.pt[2]);*/
	endTime = clock();
	constnTime += (endTime-startTime)/CLOCKS_PER_SEC;
	
	/*if(procRank == 0)
		printf("Computing Cofm done\n");*/
	startTime=clock();
#if 1
	long int totalVertices, numTopVertices=0;
	all_reduce(communicator(pg),labelCount, totalVertices, std::plus<long int>());
	SPIRITHelper_CountSubtreeNodes_Oct(rootNode, numTopVertices);
	totalVertices -= numTopVertices * (numProcs-1);
	float curRep = numTopVertices/(float)totalVertices * 100;
	float minOcc, occupancy = labelCount/(float)totalVertices*100;
	all_reduce(communicator(pg),occupancy, minOcc, boost::parallel::minimum<float>());
	
	long int numVerticesPendingReplication=0;
	if(minOcc < opts.replication)
	{
		numVerticesPendingReplication = totalVertices*opts.replication/100;
		numVerticesPendingReplication  -= numTopVertices;
	}
	/*if(procRank==0)
	{
		printf("replicated vertices %ld total vertices %ld\n",numTopVertices,totalVertices);	
		printf("current replication %f min occupancy %f\n",curRep,minOcc);	
		printf("numVerticesPendingReplication %ld\n", numVerticesPendingReplication);
	}*/
	/*int numLocalPseudoLeaves=0;
	long int vertexCount=0;
	std::map<long int, PLeafPointBucket>::iterator iter = pLeafMap.begin();
	for(;iter!=pLeafMap.end();iter++)
	{
		Vertex* pLeaf = reinterpret_cast<Vertex*>(iter->first);
		if(procRank == 0)
			printf("label %ld numVerticesInSubtree %ld\n",pLeaf->label, iter->second.numVerticesInSubtree);
		if(iter->second.subtreeOwner==procRank)
		{
			for(int i=0;i<pLeaf->pChild.size();i++)
			{
				long int numBottomVertices=0;
				Vertex* pRoot = pLeaf->pChild[i];
				if(pRoot)
				{
					SPIRITHelper_CountSubtreeNodes_Oct(pRoot, numBottomVertices);
					vertexCount += numBottomVertices;
				}
			}
			numLocalPseudoLeaves++;
		}
		
	}
	synchronize(pg);
	//printf("total vertices %ld\n", vertexCount+numTopVertices);
	printf("rank %d numpseudoleaves %d local_pseudoleaves %d num_bottom_level_vertices %ld numvertices %ld \n",procRank, pLeafMap.size(),numLocalPseudoLeaves, vertexCount, labelCount);*/

	//SPIRITHelper_ReplicateSubtrees(numSubtreesToBeReplicated);
	SPIRITHelper_ReplicateSubtrees_Oct(numVerticesPendingReplication);
#endif
	endTime = clock();
	constnTime += (endTime-startTime)/CLOCKS_PER_SEC;
	double maxConstnTime;
	reduce(communicator(pg),constnTime, maxConstnTime, boost::parallel::maximum<double>(),0);
	if(procRank == 0)
	{
		printf("tree construction time:%f seconds\n",maxConstnTime);
	}
	pLeafMap.erase(pLeafMap.begin(),pLeafMap.end());
	return dia;
}

/*int SPIRIT::ConstructBinaryTree(char* fileName, long int numPoints, InputFileParser* inputParser)
{
	Vec center;
	double dia, startTime, endTime, constnTime;
	long int totalPointsRead = 0;
	int step=0;
	bool done = false;

	AllocateWorkspace(numPoints, inputParser);
	
	while(totalPointsRead < numPoints)
	{
		int numRead = ReadTreeData_Oct(fileName, totalPointsRead, numPoints, address, inputParser, step);
		if(numRead == 0)
		{
			printf("ERROR. Tree Construction failed\n");
			return 0;
		}
		startTime=clock();
		step++;
		if(step == 1)
		{
			SPIRITHelper_ComputeBoundingBoxParams(address, numRead, numPoints, center, dia);
			if(!rootNode)
			{
				rootNode = CreateVertex(VTYPE_NORMAL);
				KdtreeVertexData2* vData = new KdtreeVertexData2();
				rootNode->vData = vData;
				memcpy(vData->center,center.pt,sizeof(float)*DIMENSION);
			}
			if(numPoints > opts.batchSize)
				continue;
		}
		totalPointsRead += numRead;
		int donebuildsubtree = BuildSubTree2(address,NULL, 0, numRead-1, 0, false, curTreeType, true);
		if(donebuildsubtree == BUILDSUBTREE_SAMEPROCESS)
			done=true;
		endTime = clock();
		constnTime += (endTime-startTime)/CLOCKS_PER_SEC;
	}



}*/

int SPIRIT::ConstructBinaryTree(std::vector<Point*>& points, long int numPoints, InputFileParser* inputParser)
{
	bool done = false;
	Vertex* rNode = NULL;
	MsgUpdatePLeaves_Oct msgUpdatePLeaves;
	int expectedUpdates=0;

	numTraversals = numPoints;
	parser=inputParser;
	curTreeType = parser->GetTreeType();
	AllocateWorkspace(numPoints, inputParser);
	traversalsTBD = ReadTreeData_Oct(NULL, 0, numPoints, address, inputParser, 1); //a placeholder till iterative version of ConstructBinaryTree is developed.

	double maxConstnTime, constnTime=0, startTime = clock();
	rootNode = CloneSubTree(points,NULL,0,points.size(),0);
	if(!remotePseudoLeavesPresent)
		done = true;

	if(!done)
	{
		for(int i=0;i<rank0PseudoLeaves.size();i++)
		{
			Vertex* pLeaf = rank0PseudoLeaves[i];
			MsgUpdatePLeaf_Oct m;
			m.descs.push_back(procRank);
			if(pLeaf->leftChild)
			{
				m.cell.push_back(0);
				m.children.push_back(reinterpret_cast<long int>(pLeaf->leftChild));
			}
			if(pLeaf->rightChild)
			{
				m.cell.push_back(1);
				m.children.push_back(reinterpret_cast<long int>(pLeaf->rightChild));
			}
			m.label = pLeaf->label;
			m.numVertices=0;
			pLeaf->level++; 
			SPIRITHelper_CountSubtreeNodes(pLeaf, m.numVertices);
			pLeafMap[reinterpret_cast<long int>(pLeaf)].numVerticesInSubtree = m.numVertices;
			pLeaf->level--; 
			msgUpdatePLeaves.vPLeaves.push_back(m);
		}
		
		if(procRank == 0)
		{
			std::map<int, int> uniqBottomLevelNodes;
			std::map<long int,PLeafPointBucket>::iterator iter=pLeafMap.begin();
			for(;iter!=pLeafMap.end();iter++)
			{
				std::pair<std::map<int, int>::iterator,bool> isUniq = uniqBottomLevelNodes.insert(std::make_pair(iter->second.subtreeOwner, 1));
				if(isUniq.second)
					expectedUpdates++;
			}
			expectedUpdates--;
		}
		else
			expectedUpdates = 1;


		if(msgUpdatePLeaves.vPLeaves.size() > 0)
		{
			if(procRank !=0)
			{
				SendMessage(0,MESSAGE_UPDATE_PLEAVES_OCT,&msgUpdatePLeaves); 
			}
			msgUpdatePLeaves.vPLeaves.clear();
		}
	}
	
	PBGL_oobMessage msg;
	while (!done)
	{
		//poll for messages
		PBGL_AsyncMsg pollMsg = pg.poll();
		if(!pollMsg)
		{
			continue;
		}
		else
			msg = pollMsg.get();

		switch(msg.second)
		{
			case MESSAGE_UPDATE_PLEAVES_OCT:
			{
					MsgUpdatePLeaves_Oct msgUpdatePLeaves;
					receive_oob(pg, msg.first, msg.second, msgUpdatePLeaves);
					int status = SPIRITHelper_UpdatePLeaves_Oct(msgUpdatePLeaves);
					expectedUpdates--;
					if(expectedUpdates == 0)
					{
						if(procRank == 0)
						{
							MsgUpdatePLeaves_Oct msgUpdatePLeaves;
							std::map<long int,PLeafPointBucket>::iterator iter=pLeafMap.begin();
							for(;iter!=pLeafMap.end();iter++)
							{
								Vertex* pLeaf = reinterpret_cast<Vertex*>(iter->first);
								MsgUpdatePLeaf_Oct m;
								m.descs.push_back(-1);
								if(pLeaf->leftChild)
								{
									m.descs[0]=pLeaf->leftDesc;
									m.cell.push_back(0);
									m.children.push_back(reinterpret_cast<long int>(pLeaf->leftChild));
								}
								if(pLeaf->rightChild)
								{
									m.descs[0]=pLeaf->rightDesc;
									m.cell.push_back(1);
									m.children.push_back(reinterpret_cast<long int>(pLeaf->rightChild));
								}
								m.label = pLeaf->label;
								msgUpdatePLeaves.vPLeaves.push_back(m);
							}

							for(int i=1;i<numProcs;i++)
							{
								SendMessage(i,MESSAGE_UPDATE_PLEAVES_OCT,&msgUpdatePLeaves); 
							}
						}	
						done = true;
					}
				}
			default:break;
		}
		
	}

	long int totalVertices, numTopVertices=0;
	all_reduce(communicator(pg),labelCount, totalVertices, std::plus<long int>());
	SPIRITHelper_CountSubtreeNodes(rootNode, numTopVertices);
	totalVertices -= numTopVertices * (numProcs-1);
	float curRep = numTopVertices/(float)totalVertices * 100;
	float minOcc, occupancy = labelCount/(float)totalVertices*100;
	all_reduce(communicator(pg),occupancy, minOcc, boost::parallel::minimum<float>());
	
	long int numVerticesPendingReplication=0;
	if(minOcc < opts.replication)
	{
		numVerticesPendingReplication = totalVertices*opts.replication/100;
		numVerticesPendingReplication  -= numTopVertices;
	}
	double endTime = clock();
	constnTime+=(endTime-startTime)/CLOCKS_PER_SEC;
	if(procRank==0)
	{
		printf("replicated vertices %ld total vertices %ld\n",numTopVertices,totalVertices);	
		printf("current replication %f min occupancy %f\n",curRep,minOcc);	
		printf("numVerticesPendingReplication %ld\n", numVerticesPendingReplication);
	}
	startTime = clock();
	SPIRITHelper_ReplicateSubtrees_Oct(numVerticesPendingReplication);
	pLeafMap.erase(pLeafMap.begin(),pLeafMap.end());
	endTime = clock();
	constnTime+=(endTime-startTime)/CLOCKS_PER_SEC;
	reduce(communicator(pg),constnTime, maxConstnTime, boost::parallel::maximum<double>(),0);
	if(procRank == 0)
		printf("tree construction time:%f seconds\n",maxConstnTime);

	//printf("LabelCount:%d\n",labelCount);
	/*if(procRank == 0)
	{
	FILE* fp = fopen("treelog2.txt","w+");
	print_treetofile(fp);
	fclose(fp);
	}*/

	
return 0;
}


Vertex* SPIRIT::CloneSubTree(std::vector<Point*>& points, Vertex* subtreeRoot, int from, int to, int depth)
{
	if(to <= from)
		return NULL;

	int median = ( from + to) / 2;
	int newStartIndex = from;
		
	Vertex* intNode; 
	if((depth==opts.subtreeHeight) || (depth==0))
	{
		intNode = CreateVertex(VTYPE_PSEUDOROOT);
		intNode->pseudoRoot = true;
	}
	else	
		intNode = CreateVertex(VTYPE_NORMAL);
	
	intNode->level = depth;
	intNode->parentDesc = procRank;
	intNode->parent = subtreeRoot;

	
	if((to - from) <= opts.maxPointsPerCell)
	{
		intNode->leaf=true;
		intNode->numPointsInCell = to-from;
		if(curTreeType == VPTREE)
		{
			for(int i=from;i<to;i++)
			{
				VptreeVertexData* tmpvData = new VptreeVertexData(0.0,(VptreePoint*)(points[i]));
				tmpvData->next = intNode->vData;
				intNode->vData = (VptreeVertexData*)tmpvData;
			}
		}
		else if((curTreeType == KDTREE)||(curTreeType == PCKDTREE))
		{
			for(int i=from;i<to;i++)
			{
				KdtreeVertexData* tmpvData = new KdtreeVertexData(points[i]);
				tmpvData->next = intNode->vData;
				intNode->vData = (KdtreeVertexData*)tmpvData;
			}
		}
	}
	else
	{
		if((depth == opts.subtreeHeight) && subtreeRoot)
		{
			
			int nextprocessid = SPIRITHelper_GetNextProcessId(reinterpret_cast<long int>(subtreeRoot));
			if(nextprocessid != procRank)
			{
				remotePseudoLeavesPresent = true;
				delete intNode;
				intNode = NULL;
				return intNode;
			}
		}

		if(curTreeType == VPTREE)
		{
			DistanceComparator* comparator = new DistanceComparator((points[from]));
			my_nth_element(points,from+1,(to-from)/2, to-1, comparator, curTreeType); 
			delete comparator;
			float tmpThreshold = mydistance((points[from]), (points[median]));
			VptreeVertexData* vData =new VptreeVertexData(tmpThreshold,(VptreePoint*)(points[from]));
			intNode->vData = vData;
			if(subtreeRoot)
			{
				vData->parent = ((VptreeVertexData*)subtreeRoot->vData)->p; 
				vData->parentThreshold = ((VptreeVertexData*)subtreeRoot->vData)->threshold; 
			}
			newStartIndex = from+1;
		}
		else if((curTreeType == KDTREE)||(curTreeType == PCKDTREE))
		{
			SplitDimComparator* comparator = new SplitDimComparator(depth%DIMENSION);
			my_nth_element(points,from+1,(to-from)/2, to-1, comparator, curTreeType); 
			float splitVal = points[median]->pt[depth%DIMENSION];
			float min[DIMENSION],max[DIMENSION], center[DIMENSION];
			for(int i=0;i<DIMENSION;i++)
			{
				min[i]=FLT_MAX;
				max[i]=-FLT_MAX;
			}
			for (long int i = from; i < to; i++) 
			{
				for(int j=0;j<DIMENSION;j++)
				{
					/* compute bounding box */
					if(min[j] > points[i]->pt[j])
						min[j] = points[i]->pt[j];
					if(max[j] < points[i]->pt[j])
						max[j] = points[i]->pt[j];
				}
			}
			float boxSum=0.0;
			for(int j=0;j<DIMENSION;j++)
			{
				center[j]=(min[j]+max[j])/2;
				float boxDist = (max[j] - min[j])/2;
				boxSum += boxDist*boxDist;
			}
			
			intNode->vData = new KdtreeVertexData(center,splitVal,sqrt(boxSum));
		}

		intNode->leftChild = CloneSubTree(points,intNode,newStartIndex, median,depth+1);
		intNode->leftDesc = procRank;
		if((curTreeType == VPTREE) && intNode->leftChild)
			((VptreeVertexData*)intNode->leftChild->vData)->isLeftChild = true;
		intNode->rightChild = CloneSubTree(points,intNode,median, to,depth+1);
		intNode->rightDesc = procRank;
		if((curTreeType == VPTREE) && intNode->rightChild)
			((VptreeVertexData*)intNode->rightChild->vData)->isLeftChild = false;
	}

	if(depth==opts.subtreeHeight-1)
	{
		intNode->pseudoLeaf = true;
		intNode->label = uniqPLeafLabel++;
		if((intNode->leftChild && !intNode->leftChild->leaf) || (intNode->rightChild && !intNode->rightChild->leaf))
			rank0PseudoLeaves.push_back(intNode);
	}

	return intNode;
	
}

void SPIRIT::FreeWorkspace()
{
	if(shm)
	{
		shm->remove("SW1");
		delete sharedWorkspace;
		delete shm;
		shm=NULL;
		sharedWorkspace=NULL;
		address = NULL;
	}
}

void SPIRIT::AllocateWorkspace(long int numPoints, InputFileParser* inputParser)
{
	unsigned long totSize=numPoints;
	int offsetFactor=0;


	if(curTreeType == OCTREE)
		offsetFactor = sizeof(OctreePoint);
	else if(curTreeType == KDTREE)
	{
		offsetFactor = sizeof(KdtreePoint);
		totSize -=  totSize/2;
	}
	else if(curTreeType == PCKDTREE)
		offsetFactor = sizeof(PCKdtreePoint);
	else if(curTreeType == VPTREE)
	{
		offsetFactor = sizeof(VptreePoint);
		totSize -=  totSize/2;
	}

	if(totSize > opts.batchSize)
		totSize = opts.batchSize;
	totSize *= offsetFactor;

	using namespace boost::interprocess;
	//Create a shared memory object.
	//Map the whole shared memory in this process
	shm = new shared_memory_object(open_or_create, "SW1", read_write);
	shm->truncate(totSize);
	sharedWorkspace  = new mapped_region(*shm, read_write);
	address = (unsigned char*)(sharedWorkspace->get_address());
	//if(procRank < 9)
		memset(address,0,totSize);
	return;
}

void SPIRIT::SPIRITHelper_ComputeBoundingBoxParams(const unsigned char* bodies, long int size, long int numPoints, Vec& center, float& dia)
{
		
	Vec pos;
	Vec max(-FLT_MAX), min(FLT_MAX);
	int offsetFactor = sizeof(OctreePoint);
	const unsigned char* nbodies = bodies;//reinterpret_cast<const unsigned char *>(bodies);

	/* compute the max and min positions to form a bounding box */
	for (int i=0; i<size; i++ ) 
	{
		const OctreePoint* p = reinterpret_cast<const OctreePoint*>(nbodies+i*offsetFactor);
		for(int i=0;i<DIMENSION;i++)
		{
			if(min.pt[i] > p->cofm.pt[i])
				min.pt[i] = p->cofm.pt[i];
			if(max.pt[i] < p->cofm.pt[i])
				max.pt[i] = p->cofm.pt[i];
		}
	}

	if(numPoints > size)
	{
		Vec globalMin, globalMax;
		reduce(communicator(pg),min.pt[0], globalMin.pt[0], mpi::minimum<float>(),0);
		reduce(communicator(pg),min.pt[1], globalMin.pt[1], mpi::minimum<float>(),0);
		reduce(communicator(pg),min.pt[2], globalMin.pt[2], mpi::minimum<float>(),0);
		reduce(communicator(pg),max.pt[0], globalMax.pt[0], boost::parallel::maximum<float>(),0);
		reduce(communicator(pg),max.pt[1], globalMax.pt[1], boost::parallel::maximum<float>(),0);
		reduce(communicator(pg),max.pt[2], globalMax.pt[2], boost::parallel::maximum<float>(),0);
			
		float diameter=(globalMax.pt[0]-globalMin.pt[0]);
		if ((diameter<(globalMax.pt[1]-globalMin.pt[1])))
		{
			diameter=(globalMax.pt[1]-globalMin.pt[1]);
		}
		if ((diameter<(globalMax.pt[2]-globalMin.pt[2])))
		{
			diameter=(globalMax.pt[2]-globalMin.pt[2]);
		}
			
		broadcast(communicator(pg),diameter,0);
		dia = diameter;
		center.pt[0]=((globalMax.pt[0]+globalMin.pt[0])*0.5);
		center.pt[1]=((globalMax.pt[1]+globalMin.pt[1])*0.5);
		center.pt[2]=((globalMax.pt[2]+globalMin.pt[2])*0.5);
		broadcast(communicator(pg),center.pt[0],0);
		broadcast(communicator(pg),center.pt[1],0);
		broadcast(communicator(pg),center.pt[2],0);
	}
	else
	{
		dia = max.pt[0]-min.pt[0];
		/* compute the maximum of the diameters of all axes */
		for(int i=1;i<DIMENSION;i++)
		{
			if (dia<(max.pt[i]-min.pt[i]))
			{
				dia=max.pt[i]-min.pt[i];
			}
		}

		/* compute the center point */
		for(int i=0;i<DIMENSION;i++)
		{
			center.pt[i] = (max.pt[i] + min.pt[i])/2;
		}
	}
	return;

}

void SPIRIT::SPIRITHelper_ComputeBoundingBoxParams(const TPointVector& nbodies, Vec& center, float& dia)
{
	Vec pos;
	Vec max(-FLT_MAX), min(FLT_MAX);

	/* compute the max and min positions to form a bounding box */
	for (int i=0; i<nbodies.size(); i ++ ) 
	{
		OctreePoint* p = (OctreePoint*)(nbodies[i]);
		for(int i=0;i<DIMENSION;i++)
		{
			if(min.pt[i] > p->cofm.pt[i])
				min.pt[i] = p->cofm.pt[i];
			if(max.pt[i] < p->cofm.pt[i])
				max.pt[i] = p->cofm.pt[i];
		}
	}

	dia = max.pt[0] - min.pt[0];
	/* compute the maximum of the diameters of all axes */
	for(int i=1;i<DIMENSION;i++)
	{
		if (dia<(max.pt[i]-min.pt[i]))
		{
			dia=max.pt[i]-min.pt[i];
		}
	}

	/* compute the center point */
	for(int i=0;i<DIMENSION;i++)
	{
		center.pt[i] = (max.pt[i] + min.pt[i])/2;
	}
	return;

}

int SPIRIT::SPIRITHelper_ComputeCofm(Vertex* node, bool explorePseudoLeafChildren)
{
		int ret = STATUS_TRAVERSE_COMPLETE;
		if(node->leaf)
		{
			VertexData* tmpVData= node->vData;
			OctreeVertexData* vData;
			float mass=0.;
			Vec cofm(0.0);
			while(tmpVData)
			{
				vData = ((OctreeVertexData*)(tmpVData));
				mass +=vData->p->mass;
				for(int i=0;i<DIMENSION;i++)
					cofm.pt[i] += (vData->p->cofm.pt[i] * vData->p->mass);
				tmpVData = tmpVData->next;
			}
			for(int i=0;i<DIMENSION;i++)
				cofm.pt[i] /= mass;
			vData= ((OctreeVertexData*)(node->vData));
			vData->mass = mass;
			vData->cofm = cofm;
			return STATUS_TRAVERSE_COMPLETE;
		}

		float tmpmass=0;
		Vec tmpcofm(0.0);
		//Initialize cofm to 0.		
		if(explorePseudoLeafChildren)
		{
			if(((OctreeVertexData*)(node->vData))->mass != 0)
			{
				tmpmass =((OctreeVertexData*)(node->vData))->mass; 
				tmpcofm =((OctreeVertexData*)(node->vData))->cofm; 
			}
			Vec cofm(0.0f);
			((OctreeVertexData*)(node->vData))->cofm = cofm;
			((OctreeVertexData*)(node->vData))->mass = 0.0f;
		}
		else
		{
			if((numProcs>1) && (node->level == opts.subtreeHeight-1))
				return ret;
		}
	
		float mass=0;
		Vec cofm(0.0);
		/*float tmpmass, mass = ((OctreeVertexData*)(node->vData))->mass;
		Vec tmpcofm, cofm = ((OctreeVertexData*)(node->vData))->cofm;
		if(node->level == opts.subtreeHeight-1)
		{
			tmpmass=mass;
			mass=0;
			tmpcofm=cofm;
			cofm.pt[0]=0;cofm.pt[1]=0;cofm.pt[2]=0;
		}*/
		assert(node->pChild.size() == 8);
		for(int i=0;i<node->pChild.size();i++)
		{
			if(node->pChild[i])
			{
				if(node->childDesc[i] != procRank) 
				{
					continue;	
				}
				int status = SPIRITHelper_ComputeCofm(node->pChild[i], explorePseudoLeafChildren);
				if(status == STATUS_TRAVERSE_COMPLETE)
				{
					float childMass = ((OctreeVertexData*)(node->pChild[i]->vData))->mass;
					const Vec& childCofm = ((OctreeVertexData*)(node->pChild[i]->vData))->cofm;
					for(int i=0;i<DIMENSION;i++)
						cofm.pt[i] += childCofm.pt[i] * childMass;
					mass += childMass;
				}
				else
				{
					ret = status;
					break;
				}
			}
		}
		if(ret == STATUS_TRAVERSE_COMPLETE)	
		{
			for(int i=0;i<DIMENSION;i++)
				cofm.pt[i] /= mass;
			//printf("%d 0 %d %f %f %f %f\n",node->id, node->vData->level, ((BH_Vertex*)node)->mass, ((BH_Vertex*)node)->cofm.x, ((BH_Vertex*)node)->cofm.y, ((BH_Vertex*)node)->cofm.z); 
		}
		
		/*if(tmpmass != 0)
		{
			if(tmpmass != mass)
				printf("ERROR. PLeaf ID:%ld Mass Not correct during replication old (%f) new (%f)\n",node->label, tmpmass, mass);
			if((tmpcofm.pt[0] != cofm.pt[0]) || (tmpcofm.pt[1] != cofm.pt[1]) || (tmpcofm.pt[2] != cofm.pt[2]))
				printf("ERROR. Pleaf ID:%ld Cofm not correct during replication old (%f %f %f) new (%f %f %f)\n", node->label, tmpcofm.pt[0],tmpcofm.pt[1],tmpcofm.pt[2],cofm.pt[0],cofm.pt[1],cofm.pt[2]);
		}*/
		((OctreeVertexData*)(node->vData))->mass = mass;
		((OctreeVertexData*)(node->vData))->cofm = cofm;
	return ret;
}

int SPIRIT::BuildSubTree_Oct(Vertex* subtreeRoot, int rOwner, Point* point, Vec& center, float dia, int depth, int DOR, bool clonePoint)
{
	int ret = BUILDSUBTREE_SAMEPROCESS;
	assert(subtreeRoot != NULL);
	
	int space=0;
	Vec offset(0.0f);

	if (center.pt[0] < ((OctreePoint*)point)->cofm.pt[0]) 
	{
		space += 1;
		offset.pt[0] = dia;
	}
	if (center.pt[1] < ((OctreePoint*)point)->cofm.pt[1]) 
	{
		space += 2;
		offset.pt[1] = dia;
	}
	if (center.pt[2] < ((OctreePoint*)point)->cofm.pt[2]) 
	{
		space += 4;
		offset.pt[2] = dia;
	}

	Vertex *child = subtreeRoot->pChild[space];
	if (child == NULL)
	{
		if(depth==opts.subtreeHeight)
		{
			for(int i=0;i<rank0PseudoLeaves.size();i++)
			{
				if(subtreeRoot == rank0PseudoLeaves[i])
				{
					DOR=0;
					break;
				}
			}
		}
		if((numProcs==1) || (depth != opts.subtreeHeight) || ((depth == opts.subtreeHeight) && (DOR==0)))
		{
			Vertex *n=NULL;
			if(DOR == 0) //this part of the code is added only to collect traversal profile. Strictly, it is not necessary to have vType=PSEUDOROOT when only single process is present.
			{
				n = (Vertex *)CreateVertex(VTYPE_PSEUDOROOT);
				n->pseudoRoot = true;
			}
			else
				n = (Vertex *)CreateVertex(VTYPE_NORMAL);
			n->vData = new OctreeVertexData((OctreePoint*)point, ((OctreePoint*)point)->mass, ((OctreePoint*)point)->cofm, clonePoint);
			n->leaf = true;
			n->level = depth;
			if(depth  == opts.subtreeHeight-1)
				n->label = uniqPLeafLabel++;
			n->numPointsInCell++;
			subtreeRoot->pChild[space]=n;
			subtreeRoot->childDesc[space]=procRank;
			if(rOwner != procRank)
				n->parent = subtreeRoot->parent;
			else
				n->parent = subtreeRoot;
			n->parentDesc = rOwner;
			ret = BUILDSUBTREE_SAMEPROCESS;
		}
	}
	else 
	{
			if (child->leaf)
			{
				OctreeVertexData* vData = (OctreeVertexData*)child->vData;
				if(vData->p->cofm == ((OctreePoint*)point)->cofm)
				{
					Vec& lc = vData->p->cofm;
					Vec& rc = ((OctreePoint*)point)->cofm;
					printf("%d: ERROR. %ld (%f %f %f) replica of body %ld (%f %f %f)\n",procRank, vData->p->id,lc.pt[0],lc.pt[1],lc.pt[2],point->id,rc.pt[0],rc.pt[1],rc.pt[2]);
					//this can sometimes happen with synthetic data.
					//printf("Points at same position detected!\n"); 
					//vData->mass += ((OctreePoint*)point)->mass;
					return ret;
				}
				
				if(child->numPointsInCell == opts.maxPointsPerCell)
				{
					//OctreePoint* childPoint = vData->p;
					child->leaf = false;
					child->numPointsInCell=0;
					float halfR = 0.5f * dia;
					Vec newCenter;
					for(int i=0;i<DIMENSION;i++)
						newCenter.pt[i] = (center.pt[i]-halfR)+offset.pt[i];
					if((numProcs > 1) && (child->level == (opts.subtreeHeight - 1)))
					{
						long int pLeaf = reinterpret_cast<long int>(child);
						int nextprocessid = SPIRITHelper_GetNextProcessId(pLeaf);
						if(nextprocessid != procRank)
						{
							child->pseudoLeaf=true;
							OctreeVertexData* otherVData = (OctreeVertexData*)(vData);
							int i=0;
							while(otherVData)
							{
								i++;
								OctreeVertexData* nextVData=(OctreeVertexData*)(otherVData->next);
								if(i!=1)
									delete otherVData;
								otherVData = nextVData;
							}
							return BUILDSUBTREE_MAXHEIGHT;
						}
						else
						{
							rank0PseudoLeaves.push_back(child);
							float * tmp = new float[4];
							tmp[0]=newCenter.pt[0];
							tmp[1]=newCenter.pt[1];
							tmp[2]=newCenter.pt[2];
							tmp[3]=halfR;
							rank0PseudoLeavesParams.push_back(tmp);
						}
					}
					OctreeVertexData* otherVData = (OctreeVertexData*)(vData);
					while(otherVData)
					{
						OctreeVertexData* nextVData = (OctreeVertexData*)(otherVData->next);
						OctreePoint* cPoint = otherVData->p;
						int childNum = SPIRITHelper_ComputeChildNumber(newCenter, cPoint->cofm);
						if(child->pChild[childNum] == NULL)
						{
							otherVData->next=NULL;
							Vertex* newNode;
							if((numProcs > 1) && (child->level == opts.subtreeHeight-1))
							{
								newNode = (Vertex *)CreateVertex(VTYPE_PSEUDOROOT);
								newNode->pseudoRoot = true;
							}
							else
								newNode = (Vertex *)CreateVertex(VTYPE_NORMAL);
							newNode->leaf = true;
							newNode->level = child->level + 1;
							if(child->level  == opts.subtreeHeight-2)
								newNode->label = uniqPLeafLabel++;
							newNode->vData = otherVData;
							newNode->numPointsInCell++;
							child->pChild[childNum] = newNode;
							child->childDesc[childNum] = procRank;
							newNode->parentDesc = procRank;
							newNode->parent = child;
						}
						else
						{
							otherVData->p=NULL; //to avoid deleting the containing point, which will be inserted into a new cell (below).
							delete otherVData;
							ret = BuildSubTree_Oct(child,procRank,cPoint,newCenter, halfR,depth+1, DOR+1, false);
						}
						otherVData = nextVData;
					}
					ret = BuildSubTree_Oct(child,procRank,point,newCenter, halfR,depth+1, DOR+1, clonePoint);
					child->vData = new OctreeVertexData();
				}
				else
				{
					OctreeVertexData* newvData = new OctreeVertexData((OctreePoint*)point, ((OctreePoint*)point)->mass, ((OctreePoint*)point)->cofm, clonePoint);
					newvData->next = child->vData;
					child->vData = newvData;
					child->numPointsInCell++;
				}
			}
			else
			{
				float halfR = 0.5f * dia;
				Vec newCenter;
				for(int i=0;i<DIMENSION;i++)
					newCenter.pt[i] = (center.pt[i]-halfR)+offset.pt[i];
				ret = BuildSubTree_Oct(child,procRank,point,newCenter, halfR,depth+1, DOR+1, clonePoint);
			}
	}

	return ret;
}

int SPIRIT::SPIRITHelper_ComputeChildNumber(const Vec& cofm, const Vec& chCofm)
{
	int numChild = 0;
	if(cofm.pt[0] < chCofm.pt[0])
	{
		numChild = 1;
	}
	if(cofm.pt[1] < chCofm.pt[1]) 
	{
		numChild += 2;
	}
	if(cofm.pt[2] < chCofm.pt[2]) 
	{
		numChild += 4;
	}
	return numChild;
}

void SPIRIT::SPIRITHelper_ReplicateSubtrees_Oct(long int numVerticesPendingReplication)
{
	PBGL_oobMessage msg;
	bool done=false;
	std::vector<MsgBuildSubTree*> msgBox;
	int numAcksExpected = 0;
	long int numReqs = numVerticesPendingReplication;
	int dbgtreecnt=0;
	bool readyToSend=false;
	int prevModifier=procRank;
	
	if(numReqs == 0)
		return;

	if(procRank == 0)
		readyToSend = true;
	else
		readyToSend = false;

	while (!done)
	{
		PBGL_AsyncMsg pollMsg = pg.poll();
		if(!pollMsg)
		{
			if(msgBox.size() > 0)
			{
				if(readyToSend)
				{
					readyToSend = false;
					numAcksExpected = numProcs-1;
					//printf("%d Broadcasting a subtree chunk\n",procRank);
					for(int j=0;j<numProcs;j++)
					{
						if(j != procRank)
						{
							SendMessage(j,MESSAGE_BUILDSUBTREE,msgBox[0]); 
						}
					}
					MsgBuildSubTree* msg = reinterpret_cast<MsgBuildSubTree*>(msgBox[0]);
					msg->ptv.clear();
					delete msg;
					msgBox.erase(msgBox.begin());
				}
			}
			else if(readyToSend)
			{
				Vertex* pLeaf=NULL;
				if((numReqs > 0) && (rank0PseudoLeaves.size() > 0))
				{
					pLeaf = rank0PseudoLeaves[0];
					MsgBuildSubTree msgRepSubtree;
					msgRepSubtree.type = curTreeType;
					long int numVertices=0;
					if(curTreeType == OCTREE)
					{
						for(int j=0;j<pLeaf->pChild.size();j++)
						{
							Vertex* pRoot = pLeaf->pChild[j];
							if(pRoot)
								numVertices+=SPIRITHelper_SaveSubtree(pRoot, msgRepSubtree.ptv);
						}
					}
					else
					{
						if(curTreeType == VPTREE)
						{
							numVertices = SPIRITHelper_SaveSubtree(pLeaf, msgRepSubtree.ptv);
							//decrement by one because pLeaf is already replicated on all nodes. The pLeaf data is still sent for implementation ease.
							numVertices--; 
						}
						else	
						{
							if((pLeaf->leftChild) && (pLeaf->leftDesc== procRank))
								numVertices+=SPIRITHelper_SaveSubtree(pLeaf->leftChild, msgRepSubtree.ptv);
							if((pLeaf->rightChild) && (pLeaf->rightDesc ==procRank))
								numVertices+=SPIRITHelper_SaveSubtree(pLeaf->rightChild, msgRepSubtree.ptv);
						}
					}
					msgRepSubtree.headerChunk = true;	
					msgRepSubtree.pLeafLabel = pLeaf->label;
					msgRepSubtree.numVertices = numVertices;
					if(curTreeType == OCTREE)
					{
						msgRepSubtree.center.pt[0] = rank0PseudoLeavesParams[0][0];
						msgRepSubtree.center.pt[1] = rank0PseudoLeavesParams[0][1];
						msgRepSubtree.center.pt[2] = rank0PseudoLeavesParams[0][2];
						msgRepSubtree.diameter = rank0PseudoLeavesParams[0][3];
					}
					//printf("2 Initiating new subtree replication %d pLeaf ID %ld (%d) numReqs %ld (subtree size %ld)\n",++dbgtreecnt,pLeaf->label,procRank,numReqs, msgRepSubtree.ptv.size());
					if(msgRepSubtree.ptv.size() > 5000)
					{
						//printf("Large Subtree. %d points\n",msgRepSubtree.ptv.size());
						long int totalSize = msgRepSubtree.ptv.size();
						long int startIndex = 0;
						long int endIndex = 5000;
						while(true)
						{
							MsgBuildSubTree* m = new MsgBuildSubTree();
							m->type = curTreeType;
							if(startIndex == 0)
								m->headerChunk = true;	
							m->moreData = true;
							m->pLeafLabel = msgRepSubtree.pLeafLabel;
							if(curTreeType == OCTREE)
							{
								m->diameter = msgRepSubtree.diameter; 
								m->center = msgRepSubtree.center; 
							}
							m->numVertices = msgRepSubtree.numVertices;
							if(endIndex == totalSize)
								m->moreData = false;
							m->ptv.insert(m->ptv.begin(),msgRepSubtree.ptv.begin()+startIndex,msgRepSubtree.ptv.begin()+endIndex);
							msgBox.push_back(m);
							if(endIndex == totalSize)
								break;
							startIndex = endIndex;
							endIndex = (endIndex + 5000)>=totalSize?totalSize:(endIndex+5000);
						}
						msgRepSubtree.ptv.clear();
					}
					else
					{
						readyToSend = false;
						numAcksExpected = numProcs-1;
						//printf("%d Broadcasting subtree info\n",procRank);
						for(int i=0;i<numProcs;i++)
						{
							if(i!=procRank)
							{
								SendMessage(i,MESSAGE_BUILDSUBTREE, &msgRepSubtree);
							}
						}
						msgRepSubtree.ptv.clear();
					}
					numReqs-=numVertices;
					continue;
				}

				readyToSend = false;
				int nxtProcID = procRank+1;
				if(nxtProcID == numProcs)
					nxtProcID=0;
				MsgReplicateReq msg;
				msg.pLeafLabel = prevModifier; 
				SendMessage(nxtProcID,MESSAGE_REPLICATE_REQ,&msg); 
			}
			continue;
		}
		else
		{
			msg = pollMsg.get();
			switch(msg.second)
			{
				case MESSAGE_REPLICATE_REQ:
				{
					MsgReplicateReq msgRepReq;
					receive_oob(pg, msg.first, msg.second, msgRepReq);
					readyToSend = true;
					prevModifier=msgRepReq.pLeafLabel;
					if(prevModifier == procRank)
					{
						for(int i=0;i<numProcs;i++)
						{
							done = true;
							if(i!=procRank)
							{
								int j=0;
								SendMessage(i,MESSAGE_DONETREE,&j); 
							}
						}
					}
				}
				break;
				case MESSAGE_BUILDSUBTREE: 
				{
					MsgBuildSubTree msgBuildSubTree;
				     	receive_oob(pg, msg.first, msg.second, msgBuildSubTree);
					bool found=false;
					Vertex* pLeaf=NULL;
					long int pLeafLabel = msgBuildSubTree.pLeafLabel;
					std::map<long int,PLeafPointBucket>::iterator iter = pLeafMap.begin();
					for(;iter!=pLeafMap.end();iter++)
					{
						pLeaf = reinterpret_cast<Vertex*>(iter->first);
						if(pLeaf->label == pLeafLabel)
						{
							found=true;
							break;
						}
					}
					//assert(found);
					if(!found)
					{	
						printf("%d pleaf with ID %ld not found\n",procRank,pLeafLabel);
						assert(0);	
					}
					if(msgBuildSubTree.headerChunk)
					{
						if(curTreeType == OCTREE)
						{
							for(int i=0;i<8;i++)
							{
								/*if(pLeaf->pChild[i] && pLeaf->childDesc[i]==procRank)
								{
									assert(pLeaf->pChild[i]->leaf);
								}
								else*/
									pLeaf->pChild[i]=NULL;
							}
						}
						else
						{
							//if(pLeaf->leftChild && (pLeaf->leftDesc!=procRank))	
								pLeaf->leftChild=NULL;
							//if(pLeaf->rightChild && (pLeaf->rightDesc!=procRank))
								pLeaf->rightChild=NULL;
						}
					}

					if(curTreeType == OCTREE)
					{
						for(int i=0;i<msgBuildSubTree.ptv.size();i++)
							BuildSubTree_Oct(pLeaf,msg.first,msgBuildSubTree.ptv[i],msgBuildSubTree.center,msgBuildSubTree.diameter, opts.subtreeHeight,0, false);
					}
					else
					{
						Vertex* intNode=NULL;
						intNode=CloneSubTree(msgBuildSubTree.ptv,NULL, 0, msgBuildSubTree.ptv.size(), opts.subtreeHeight);
						intNode->label = pLeaf->label;
						intNode->parent = pLeaf->parent;
						intNode->parentDesc = procRank;
						if(pLeaf->parent->leftChild == pLeaf)
							pLeaf->parent->leftChild = intNode;
						else
							pLeaf->parent->rightChild = intNode;
						pLeafMap.erase(reinterpret_cast<long int>(pLeaf));
						delete pLeaf;
						if(curTreeType == VPTREE)
						{
							VptreeVertexData* parentVData = (VptreeVertexData*)(intNode->parent->vData);
							VptreeVertexData* childVData = (VptreeVertexData*)(intNode->vData);
							childVData->parent=parentVData->p;
							childVData->parentThreshold = parentVData->threshold;
						}
						/*pLeaf->leftChild = intNode->leftChild;
						pLeaf->leftDesc = procRank;
						pLeaf->rightChild = intNode->rightChild;
						pLeaf->rightDesc = procRank;
						if((curTreeType == PCKDTREE) || (curTreeType == KDTREE))
							((KdtreeVertexData*)pLeaf->vData)->splitVal = ((KdtreeVertexData*)intNode->vData)->splitVal;
						delete intNode; */
					}
					msgBuildSubTree.ptv.clear();
					bool ack = true;
					if(!(msgBuildSubTree.moreData))
					{
						numReqs-=msgBuildSubTree.numVertices;
						//printf("%d Subtree created\n",procRank);
						if(curTreeType == OCTREE)
						{
							bool explorePseudoLeafChildren=true;
							SPIRITHelper_ComputeCofm(pLeaf, explorePseudoLeafChildren);
						}
					}
					SendMessage(msg.first, MESSAGE_DONEOCTSUBTREE_ACK,&ack);	
				}
				break;
				case MESSAGE_DONEOCTSUBTREE_ACK:
				{
					bool tmpAck;
					receive_oob(pg, msg.first, msg.second, tmpAck);
					numAcksExpected--;
					if(numAcksExpected == 0)
					{
						if(msgBox.size() > 0)
							readyToSend = true;
						else
						{
							//rank0PseudoLeaves.erase(curPLeafIter);
							rank0PseudoLeaves.erase(rank0PseudoLeaves.begin());
							if(rank0PseudoLeavesParams.size() > 0)
							{
								delete [] rank0PseudoLeavesParams[0];
								rank0PseudoLeavesParams.erase(rank0PseudoLeavesParams.begin());
							}
							readyToSend = false;
							int nxtProcID = procRank+1;
							if(nxtProcID == numProcs)
								nxtProcID=0;
							MsgReplicateReq m;
							m.pLeafLabel = procRank; //modifier ID
							SendMessage(nxtProcID,MESSAGE_REPLICATE_REQ,&m); 
						}
					}
				}
				break;
				case MESSAGE_DONETREE:
				{
					int i;
					receive_oob(pg, msg.first, msg.second, i);
					done = true;
				}
				break;
				default: break;
			}
		}
	}
	//printf("%d Exiting\n",procRank);
	synchronize(pg);
}

//assumption: there are no pseudoleaves in the subtree.
long int SPIRIT::SPIRITHelper_SaveSubtree(Vertex* pRoot, TPointVector& ptv)
{
	if(curTreeType == VPTREE)
	{
		VptreeVertexData* vData = (VptreeVertexData*)(pRoot->vData);
		while(vData)
		{
			ptv.push_back(vData->p);
			vData = (VptreeVertexData*)(vData->next);
		}
	}

	if(pRoot->leaf)
	{
		if((curTreeType == KDTREE) || (curTreeType == PCKDTREE))
		{
			KdtreeVertexData* vData = (KdtreeVertexData*)(pRoot->vData);
			while(vData)
			{
				ptv.push_back(vData->p);
				vData = (KdtreeVertexData*)(vData->next);
			}
		}
		else if(curTreeType == OCTREE)
		{
			OctreeVertexData* vData = (OctreeVertexData*)(pRoot->vData);
			while(vData)
			{
				ptv.push_back(vData->p);
				vData = (OctreeVertexData*)(vData->next);
			}
		}
		return 1;
	}
	
	if(curTreeType != OCTREE)
	{
		long int numVertices=0;
		numVertices++;
		if(pRoot->leftChild)
		{
			numVertices+=SPIRITHelper_SaveSubtree(pRoot->leftChild, ptv);
		}
		
		if(pRoot->rightChild)
		{
			numVertices+=SPIRITHelper_SaveSubtree(pRoot->rightChild, ptv);
		}
		return numVertices;
	}
	else
	{
		long int numVertices=0;
		numVertices++;
		for(int i=0;i<pRoot->pChild.size();i++)
		{
			if(pRoot->pChild[i])
			{
				numVertices +=SPIRITHelper_SaveSubtree(pRoot->pChild[i], ptv);
			}
		}
		return numVertices;
	}
}

/*int SPIRIT::SPIRITHelper_GetBottleneckSubtreesAndBroadcast(int numBottlenecks, char* bneckfile)
{
	int ret = STATUS_SUCCESS;
	std::vector<SubtreeHeader> bottlenecks;
	std::ifstream input(bneckfile, std::fstream::in);
	if(input.fail())
	{
		printf("Bottlenecks not specified. No replication.\n");
		return ret;
	}
	else
	{
		SPIRITHelper_ReadBottleneckDetails(input, numBottlenecks, bottlenecks);
	}
	numReplicatedSubtrees = bottlenecks.size(); 
	printf("Number of replicated subtrees :%d\n",numReplicatedSubtrees);

	std::vector<SubtreeHeader>::iterator subtreeIter = bottlenecks.begin();
	for(;subtreeIter!=bottlenecks.end();subtreeIter++)
	{
		MsgReplicateReq msg;
		msg.pLeafLabel=subtreeIter->pLeaf;
		msg.childNum = subtreeIter->childNum;
		msg.numSubtreesReplicated = bottlenecks.size();
		msg.pRoot=0;
		msg.pLeaf=0;
		SPIRIT_SendMessage(subtreeIter->pRootDesc,MESSAGE_REPLICATE_REQ,&msg);
		ret = STATUS_FAILURE;
	}
	input.close();
	return ret;
}



void SPIRIT::SPIRITHelper_ReadBottleneckDetails(std::ifstream& input, int numBottlenecks, std::vector<SubtreeHeader>& bottlenecks)
{
        while(true) 
	{
		int procID, childNum;
		long int pLeafLabel;
		input >> procID >> pLeafLabel >> childNum;
		SubtreeHeader hdr(0, childNum, procID);
		hdr.pLeaf = pLeafLabel;
		bottlenecks.push_back(hdr);
		if(bottlenecks.size() == numBottlenecks)
			break;
        }
}*/




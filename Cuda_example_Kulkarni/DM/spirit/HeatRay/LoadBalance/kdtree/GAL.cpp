#include "kdtree/GAL.h"
#include "util/timer.h"
#include <boost/graph/distributed/local_subgraph.hpp>

#ifdef SPAD_2
int numReplicatedSubtrees;
long int numReplicatedVertices;
std::vector<SubtreeHeader> adjacentSubtrees;
std::vector<long int> replicatedSubtreeIDs;
#endif

timer busyTime;
double traversalTime = 0;
long int labelCount;
#ifdef MERGE_DEGEN_TREES
//#define DEGENTREE_VCOUNT 100
#endif
int subtreeHeight;

#ifdef MESSAGE_AGGREGATION
int pipelineBufferTimerCount;
int numPipelineBuffers;
#define PIPELINE_BUFFER_TIMERVAL 50
#endif

std::vector<GAL_Vertex*> pseudoLeaves;
std::map<long int,int> pLeafMap;
#ifdef LOAD_BALANCE
std::vector<MsgUpdatePLeaf> vPLeaves;
#endif

template void GAL_BroadCastObject<Photon>(Photon& obj, GAL* g);
template void GAL::GAL_Test<Photon>(Photon& obj);
TriangleVector* points;
typedef ::math::Vector <float, 3> kvec;

#ifdef PERFCOUNTERS
double timeStep= ((CLOCKS_PER_SEC/(float)(1.0)));
std::vector<int> tpFinishedBlocks;
timer totTimer;
timer compTimer;
timer workloadTimer;
uint64_t workloadTime;
#endif

#ifdef TRAVERSAL_PROFILE
std::vector<long int> pipelineStage; 
#endif

typedef optional<PBGL_oobMessage> PBGL_AsyncMsg;

#ifdef TRAVERSAL_PROFILE
bool SortLongIntPairsBySecond_Decrease(const std::pair<long int, long int>& a, const std::pair<long int, long int>& b)
{
	return a.second < b.second;
}
#endif
GAL* GAL::instance = NULL;
GAL* GAL::GAL_GetInstance(mpi_process_group& prg, CommLineOpts* opts)
{
		if(instance == NULL)
			instance  = new GAL(prg,opts);

		return instance;
}

GAL_Vertex* GAL::GAL_GetRootNode()
{
	return rootNode;
}

GAL_Vertex* GAL::GAL_CreateVertex(TVertexType vType)
{
#ifdef PARALLEL_BGL
	labelCount++;
	//numVertices++;
	GAL_Vertex* node; 
	BGLVertexdesc d;
	if(vType == VTYPE_PSEUDOROOT)
	{
		node = new GAL_PseudoRoot();
/*#ifdef TRAVERSAL_PROFILE
		pipelineStage.push_back(reinterpret_cast<long int>(node));
#endif*/
	}
	else
	{
		node =  new GAL_Vertex();
	}
	d = add_vertex(*g);
	node->desc = d;
	node->id = labelCount;
	return node;
#endif	
}

void GAL::GAL_DeleteVertex(GAL_Vertex* node)
{
	//numVertices--;
	remove_vertex(node->desc, *g);
	delete node;
}

void GAL::GAL_SendMessage(int processId, int messageId, void* msg)
{
	busyTime.Stop();
#ifdef PARALLEL_BGL
	assert(processId != procRank);
	switch(messageId)
	{
		case MESSAGE_BUILDSUBTREE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgBuildSubTree *>(msg))); 
					break;

		case MESSAGE_DONESUBTREE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgUpdateMinMax *>(msg))); 
					break;
		case MESSAGE_DONEKDTREE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgTerminate *>(msg))); 
					break;
		case MESSAGE_TRAVERSE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgTraverse*>(msg))); 
					break;
		case MESSAGE_TRAVERSE_BACKWARD:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgTraverse*>(msg))); 
					break;
#ifdef LOAD_BALANCE
		case MESSAGE_REPLICATE_PLEAF:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgUpdateMinMax *>(msg))); 
					break;
		case MESSAGE_UPDATE_PLEAVES:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgUpdatePLeaves*>(msg))); 
					break;
		case MESSAGE_UPDATE_PROOT:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgUpdatePRoot*>(msg))); 
					break;
		case MESSAGE_READYTOEXIT:
					send_oob(pg,processId,messageId,*(reinterpret_cast<int*>(msg))); 
					break;
#endif
#ifdef MESSAGE_AGGREGATION
		case MESSAGE_SENDCOMPRESSED:
					send_oob(pg,processId,messageId,*(reinterpret_cast<int*>(msg))); 
					break;
#endif
#ifdef SPAD_2
		case MESSAGE_REPLICATE_REQ:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgReplicateReq*>(msg))); 
					break;
		case MESSAGE_REPLICATE_SUBTREE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgReplicateSubtree*>(msg))); 
					break;
#endif

		default:
			break;
	}
#endif
}

int GAL::GAL_ConstructKDTree(TriangleVector& pts)
{
#ifdef PARALLEL_BGL
	
	bool done = false;
	int depth = 0;
#ifdef MERGE_DEGEN_TREES
	int degenVertexCount=0;
	int degenTreeCount=0;
	int dgentree_vcount = GetOpt_DegentreeVCount();
#endif
#ifdef SPAD_2
	/*std::vector<SubtreeHeader> bottlenecks;
	int numBottlenecks = 5508;
	char *bneckfile = "bneck5508.txt";
	std::ifstream input(bneckfile, std::fstream::in);
	if(input.fail())
	{
		printf("Bottlenecks not specified. No replication.\n");
	}
	else
	{
		GALHelper_ReadBottleneckDetails(input, numBottlenecks, bottlenecks);
	}
	input.close();*/
#endif
	points = &pts;	
	subtreeHeight = GetOpt_SubtreeHeight();
#ifdef SPAD_2
	numReplicatedSubtrees = GetOpt_NumReplicatedSubtrees();
#endif
#ifdef MESSAGE_AGGREGATION
	numPipelineBuffers = GetOpt_NumPipelineBuffers();	
	//pipelineBufferTimerCount=PIPELINE_BUFFER_TIMERVAL;
#endif
	depth = 0;
	
	std::vector<int> pointRefs(pts.size());
	//TriangleVector::iterator piter=points->begin();
	kvec min (HUGE_VAL, HUGE_VAL, HUGE_VAL);
	kvec max (-HUGE_VAL, -HUGE_VAL, -HUGE_VAL);

	//for(;piter!=points->end();piter++)
	for(int i=0;i<pts.size();i++)
	{
		for(int j = 0; j < 3; ++j)// Loop over triangle vertices
		{
			for (int k = 0; k < 3; ++k)// Loop over vertex indices.
			{
				max[k] = std::max(max[k], pts[i].m_vertices[j][k]);
				min[k] = std::min(min[k], pts[i].m_vertices[j][k]);
			}
		}
		pointRefs[i] = i;
		pts[i].initializeIntersection();
	}
		
	GAL_Synchronize();
#ifndef LOAD_BALANCE
	//process 0 distributes the left and right subtrees among processes 0 and 1(if present)
	if(procRank == 0)
#endif
	{
		int donebuildsubtree = GAL_BuildSubTree(pointRefs,NULL, 0, 0, false, Box(min,max));
		if(donebuildsubtree == BUILDSUBTREE_SAMEPROCESS)
		{
#ifndef LOAD_BALANCE
			MsgTerminate msgTerminate;
			for(int i=1;i<numProcs;i++)
			{
				if(i!=procRank)
					GAL_SendMessage(i,MESSAGE_DONEKDTREE,&msgTerminate); 
			}
#endif
			done = true;
		}
		pointRefs.erase(pointRefs.begin(),pointRefs.end());
	}
#ifndef LOAD_BALANCE
	else
	{
		//just create and leave it empty. Will be populated when MESSAGE_DONETREE is received.
		rootNode = new GAL_Vertex();
	}
#endif
	
	PBGL_oobMessage msg;
	int donetreeval;
	int donebuildsubtree;
	
	while (!done)
	{
		//poll for messages
		PBGL_AsyncMsg pollMsg = pg.poll();
		if(!pollMsg)
			continue;
		else
			msg = pollMsg.get();
		switch(msg.second)
		{
			case MESSAGE_BUILDSUBTREE: 
				{
					MsgUpdateMinMax msgDoneTree;
					MsgBuildSubTree msgBuildSubTree;
				     	receive_oob(pg, msg.first, msg.second, msgBuildSubTree);
					//SetOpt_SubtreeHeight(msgBuildSubTree.subtreeHeight);
					GAL_Vertex node;
					GAL_Vertex* tmpLeaf;
					node.desc = msgBuildSubTree.subroot;
					node.parent = reinterpret_cast<GAL_Vertex*>(msgBuildSubTree.pLeaf);
					
					if((node.desc).owner !=  procRank)
						tmpLeaf = &node;
					else
					{
						tmpLeaf = reinterpret_cast<GAL_Vertex*>(msgBuildSubTree.pLeaf);
					}
#ifdef MERGE_HEIGHT1_TREES
					TIndices tmpIndices = msgBuildSubTree.ptv;
#endif
#ifdef LOAD_BALANCE
					if(msgBuildSubTree.depth == (subtreeHeight+1))
					{
						numUpdatesRequired+=(numProcs-2);
					}
#endif
					donebuildsubtree = GAL_BuildSubTree(msgBuildSubTree.ptv,tmpLeaf, msgBuildSubTree.depth, 0, msgBuildSubTree.isleft, msgBuildSubTree.box);
#ifdef SPAD_2
					if(msgBuildSubTree.isleft)
						((GAL_PseudoRoot*)tmpLeaf->leftChild)->pLeafLabel = msgBuildSubTree.pLeafLabel;
					else
						((GAL_PseudoRoot*)tmpLeaf->rightChild)->pLeafLabel = msgBuildSubTree.pLeafLabel;
#endif
					//If all the vertices of the subtree are owned by a single process, inform the caller process that subtree construction is done.
					if(donebuildsubtree == BUILDSUBTREE_SAMEPROCESS)
					{
						//construct donesubtree message to notify parent
#ifndef MERGE_DEGEN_TREES
						if(msgBuildSubTree.isleft)
						{
#ifdef MERGE_HEIGHT1_TREES
							if((tmpLeaf->leftChild->leftChild == NULL) && (tmpLeaf->leftChild->rightChild == NULL))
							{
								msgDoneTree.box = msgBuildSubTree.box;
								GAL_DeleteVertex(tmpLeaf->leftChild);
								tmpLeaf->leftChild = NULL;
								msgDoneTree.ptv = tmpIndices;
#ifdef LOAD_BALANCE
								if(msgBuildSubTree.depth == (subtreeHeight+1))
								{
									numUpdatesRequired-=(numProcs-2);
								}
#endif

							}
#endif
							msgDoneTree.pRoot = reinterpret_cast<long int>(tmpLeaf->leftChild);
							msgDoneTree.child = tmpLeaf->leftDesc;
						}
						else
						{
#ifdef MERGE_HEIGHT1_TREES
							if((tmpLeaf->rightChild->leftChild == NULL) && (tmpLeaf->rightChild->rightChild == NULL))
							{
								msgDoneTree.box = msgBuildSubTree.box;
								GAL_DeleteVertex(tmpLeaf->rightChild);
								tmpLeaf->rightChild = NULL;
								msgDoneTree.ptv = tmpIndices;
#ifdef LOAD_BALANCE
								if(msgBuildSubTree.depth == (subtreeHeight+1))
								{
									numUpdatesRequired-=(numProcs-2);
								}
#endif
							}
#endif
							msgDoneTree.pRoot = reinterpret_cast<long int>(tmpLeaf->rightChild);
							msgDoneTree.child = tmpLeaf->rightDesc;
						}
						msgDoneTree.pLeaf = msgBuildSubTree.pLeaf;
						msgDoneTree.isLeft = msgBuildSubTree.isleft;
#ifdef TRAVERSAL_PROFILE
						pipelineStage.push_back(msgDoneTree.pRoot); 
#endif
						GAL_SendMessage(msg.first,MESSAGE_DONESUBTREE, &msgDoneTree);
#else
						long int vertexCount = 0;
						GAL_Vertex* pRootCreated = msgBuildSubTree.isleft?tmpLeaf->leftChild:tmpLeaf->rightChild;
						msgDoneTree.child  = msgBuildSubTree.isleft?tmpLeaf->leftDesc:tmpLeaf->rightDesc;
						msgDoneTree.box = msgBuildSubTree.box;
						GALHelper_CountSubtreeNodes(pRootCreated,vertexCount);
						msgDoneTree.pRoot = reinterpret_cast<long int>(pRootCreated);
						msgDoneTree.pLeaf = msgBuildSubTree.pLeaf;
						msgDoneTree.isLeft = msgBuildSubTree.isleft;

#ifdef TRAVERSAL_PROFILE
						pipelineStage.push_back(msgDoneTree.pRoot); 
#endif
#ifdef SPAD_2
						//if(GALHelper_IsBottleneck(bottlenecks,pRootCreated))
						if((vertexCount < dgentree_vcount))
#else
						if((vertexCount < dgentree_vcount))
#endif
						{
#ifdef LOAD_BALANCE
								if(pRootCreated->level == (subtreeHeight+1))
								{
									numUpdatesRequired-=(numProcs-2);
								}
#endif
								GALHelper_DeleteSubtree(pRootCreated);
								msgDoneTree.ptv = tmpIndices;
#ifdef TRAVERSAL_PROFILE
								pipelineStage.pop_back(); 
#endif
						}
						GAL_SendMessage(msg.first,MESSAGE_DONESUBTREE, &msgDoneTree);
#endif 
					}//end BUILDSUBTREE_SAMEPROCESS
				}
				break;
			case MESSAGE_DONESUBTREE:
				{
					MsgUpdateMinMax msgDoneTree;
					bool isPseudoLeaf = true;
					receive_oob(pg, msg.first, msg.second, msgDoneTree);
					//Receive the node which is pseudo leaf.  A subtree of this node is fully constructed.
					GAL_Vertex* vert = reinterpret_cast<GAL_Vertex*>(msgDoneTree.pLeaf);
					GAL_Vertex* tmpVert = vert;
	#ifdef MERGE_HEIGHT1_TREES		
					if(msgDoneTree.ptv.size() > 0)
					{
	#ifdef LOAD_BALANCE
						if(tmpVert->level == subtreeHeight)
						{
							assert(procRank == 0);
							msgDoneTree.pLeaf = (tmpVert->desc).local;
							for(int i=1;i<numProcs;i++)
							{
								GAL_SendMessage(i,MESSAGE_REPLICATE_PLEAF,&msgDoneTree);
							}

							if(tmpVert->uCount == 1)
							{
								if((tmpVert->leftChild && (tmpVert->leftDesc == procRank)) || (tmpVert->rightChild && (tmpVert->rightDesc == procRank)))
								{
									std::vector<GAL_Vertex*>::iterator psiter = pseudoLeaves.begin();
									while(psiter != pseudoLeaves.end())
									{
										if(*psiter == tmpVert)
										{
											pseudoLeaves.erase(psiter);
											isPseudoLeaf = false;
											break;
										}
										else
											psiter++;
									}
									
								}
							}
						}
	#endif
						int ret = GAL_BuildSubTree(msgDoneTree.ptv,tmpVert, (tmpVert->level)+1, 0, msgDoneTree.isLeft, msgDoneTree.box);
						/*if(msgDoneTree.isLeft)
							tmpVert->leftChild->pseudoRoot = false;
						else
							tmpVert->rightChild->pseudoRoot = false;*/
						tmpVert->uCount++;
#ifdef MERGE_DEGEN_TREES						
						//for collecting statistics
						GAL_Vertex* pRootCreated = msgDoneTree.isLeft?tmpVert->leftChild:tmpVert->rightChild;
						long int vertexCount=0;
						GALHelper_CountSubtreeNodes(pRootCreated,vertexCount);
						if(vertexCount != 1)
							degenVertexCount += vertexCount;
						degenTreeCount++;
#endif
					}
					else
	#endif	
					{
#ifdef SPAD_2
						int i=msgDoneTree.isLeft?0:1;
						SubtreeHeader hdr(msgDoneTree.pRoot,i,msg.first);
						hdr.pLeaf = msgDoneTree.pLeaf;
						adjacentSubtrees.push_back(hdr);
#endif

						vert->uCount++;
							
						if(vert->uCount > 2)
							printf("NIKHIL ERRORCOUNT\n");

						if(msgDoneTree.isLeft)
						{
								
							vert->leftChild = reinterpret_cast<GAL_Vertex*>(msgDoneTree.pRoot);	
							vert->leftDesc = msgDoneTree.child;
						}
						else
						{
							vert->rightChild = reinterpret_cast<GAL_Vertex*>(msgDoneTree.pRoot);	
							vert->rightDesc = msgDoneTree.child;
						}
					}
					//repeatedly update all the nodes up the tree if update count of any of those is 2.
					if(tmpVert->uCount == 2)
					{
	#ifdef LOAD_BALANCE
						if(tmpVert->level == subtreeHeight)
						{
							if(isPseudoLeaf)
							{
								MsgUpdatePLeaf msgUpdatePLeaf;
								if((tmpVert->leftDesc) == procRank)
									msgUpdatePLeaf.pLeftChild = 0;
								else
								{
									msgUpdatePLeaf.leftDesc = tmpVert->leftDesc;
									msgUpdatePLeaf.pLeftChild = reinterpret_cast<long int>(tmpVert->leftChild);
								}
								if((tmpVert->rightDesc) == procRank)
									msgUpdatePLeaf.pRightChild = 0;
								else
								{
									msgUpdatePLeaf.rightDesc = tmpVert->rightDesc;
									msgUpdatePLeaf.pRightChild = reinterpret_cast<long int>(tmpVert->rightChild);
								}
								msgUpdatePLeaf.leafDesc = tmpVert->desc;
								vPLeaves.push_back(msgUpdatePLeaf);						
							}
						}
	#endif
						while(tmpVert)
						{
							//check if already at root node. If so initiate termination
							if(tmpVert->parent == NULL)
							{
								done = true;
								break;
							}
							//update parent if I am not pseudoroot
							else 
							{
									if(!tmpVert->pseudoRoot)
									{
										tmpVert = tmpVert->parent;
										(tmpVert->uCount)++;
										if(tmpVert->uCount > 2)
											printf("NIKHIL ERRORCOUNT 2\n");
										//if parent node has an update count of two, move up the tree.
										if(tmpVert->uCount == 2)
											continue;
										else
											break;	
									}
									else
									{
										//Send message to the owning process.
										if(tmpVert->isLeftChild)
											msgDoneTree.isLeft = true;
										else
											msgDoneTree.isLeft = false;
										msgDoneTree.pRoot = reinterpret_cast<long int>(tmpVert);
										msgDoneTree.pLeaf = reinterpret_cast<long int>(tmpVert->parent);
										msgDoneTree.child = procRank;
#ifdef TRAVERSAL_PROFILE
										pipelineStage.push_back(msgDoneTree.pRoot); 
#endif
										GAL_SendMessage((tmpVert->parentDesc),MESSAGE_DONESUBTREE,&msgDoneTree);
										break;	
									}
							}
						}		
					}

					if(done)
					{
	#ifdef LOAD_BALANCE
						MsgUpdatePLeaves msgUpdatePLeaves;
						msgUpdatePLeaves.vPLeaves.insert(msgUpdatePLeaves.vPLeaves.begin(),vPLeaves.begin(), vPLeaves.end());
						//printf("%d NUM_PSEUDOLEAVES %d (%d)\n",procRank,vPLeaves.size(), pseudoLeaves.size());
						for(int i=1;i<numProcs;i++)
						{
							GAL_SendMessage(i,MESSAGE_UPDATE_PLEAVES,&msgUpdatePLeaves); 
						}	
						vPLeaves.erase(vPLeaves.begin(),vPLeaves.end());

	#else

						MsgTerminate msgTerminate;
						msgTerminate.root = tmpVert->desc;
						msgTerminate.leftChild = tmpVert->leftDesc;
						msgTerminate.rightChild = tmpVert->leftDesc;
						msgTerminate.pLeft = reinterpret_cast<long int>(tmpVert->leftChild);
						msgTerminate.pRight = reinterpret_cast<long int>(tmpVert->rightChild);
						for(int i=1;i<numProcs;i++)
						{
							GAL_SendMessage(i,MESSAGE_DONEKDTREE,&msgTerminate); 
						}
	#endif
					}
				}
				break;
			case MESSAGE_DONEKDTREE:
				{
				MsgTerminate msgTerminate;
		              	receive_oob(pg, msg.first, msg.second, msgTerminate);
				done = true;
				}
				break;
#ifdef LOAD_BALANCE
			case MESSAGE_UPDATE_PLEAVES:
				{
					MsgUpdatePLeaves msgUpdatePLeaves;
					receive_oob(pg, msg.first, msg.second, msgUpdatePLeaves);
					int status = GAL_Aux_UpdatePLeaves(msgUpdatePLeaves, procRank);
					if(status == STATUS_SUCCESS)
						done = true;
					vPLeaves.erase(vPLeaves.begin(),vPLeaves.end());
					//vPLeaves.clear();
				}
				break;
			case MESSAGE_UPDATE_PROOT:
				{
					MsgUpdatePRoot msgUpdatePRoot;
					receive_oob(pg, msg.first, msg.second, msgUpdatePRoot);
					GAL_PseudoRoot* pRoot = reinterpret_cast<GAL_PseudoRoot*>(msgUpdatePRoot.pRoot);
					pRoot->parents.push_back(std::make_pair(msgUpdatePRoot.leafDesc,msgUpdatePRoot.pLeaf));
					numUpdatesRequired--;
					//printf("process:%d updates:%d\n",process_id(g->process_group()), numUpdatesRequired);
					if(GAL_Aux_IsReadyToExit())
						done = true;
				}
				break;
#ifdef MERGE_HEIGHT1_TREES
			case MESSAGE_REPLICATE_PLEAF:
				{
					MsgUpdateMinMax msgDoneTree;
					receive_oob(pg, msg.first, msg.second, msgDoneTree);
					GAL_Vertex* tmpVert = GAL_GetLocalVertex(msgDoneTree.pLeaf);
					int ret = GAL_BuildSubTree(msgDoneTree.ptv,tmpVert, (tmpVert->level)+1, 0, msgDoneTree.isLeft, msgDoneTree.box);
					/*if(msgDoneTree.isLeft)
						tmpVert->leftChild->pseudoRoot = false;
					else
						tmpVert->rightChild->pseudoRoot = false;*/
					assert(ret == BUILDSUBTREE_SAMEPROCESS);
				}
				break;
#endif
#endif
			default:break;
		}
		
	}
	
	/*std::ofstream fp;
	fp.open("treelog2.txt",std::ofstream::out);
	print_treetofile(fp);
	fp.close();*/
	
	GAL_Synchronize();
#ifdef SPAD_2
	if(procRank == 0)
	{
		if(clopts)
			clopts->PrintOpts();
	}
	GALHelper_ReplicateSubtrees();
	adjacentSubtrees.erase(adjacentSubtrees.begin(),adjacentSubtrees.end());
	numReplicatedVertices=0;
#endif
	pseudoLeaves.erase(pseudoLeaves.begin(),pseudoLeaves.end());

#ifdef MERGE_DEGEN_TREES
	int totalDegenVertexCount = 0;
	reduce(communicator(pg),degenVertexCount, totalDegenVertexCount, std::plus<int>(),0);
	int totalDegenTreeCount = 0;
	reduce(communicator(pg),degenTreeCount, totalDegenTreeCount, std::plus<int>(),0);
#endif
	/*if(degenVertexCount > 0)
		printf("%d: DegenVertexCount: %d \n",procRank,degenVertexCount);*/
	if(procRank == 0)
	{
#ifdef LOAD_BALANCE		
		long int numReplicatedVertices=0;
		GALHelper_CountSubtreeNodes(rootNode,numReplicatedVertices, false, true);
#ifdef MERGE_DEGEN_TREES
		printf("DegenTrees(< %d vertices):%d DegenVertices:%d\n",dgentree_vcount,degenTreeCount,degenVertexCount);
#endif
		printf("replicated vertices:%d\n",numReplicatedVertices);
#endif
	}
#ifdef TRAVERSAL_PROFILE


	long int totalPRootLeaves = 0;
	reduce(communicator(pg),numPRootLeaves, totalPRootLeaves, std::plus<long int>(),0);
	
	int localMaxHeight=0, maxHeightOfLeaf=0;
	std::map<int,long int>::iterator it;
	for (it=numLeavesAtHeight.begin(); it!=numLeavesAtHeight.end(); ++it)
	{
		if(it->first > localMaxHeight)
			localMaxHeight = it->first;
	}
	for(int j=0;j<numProcs;j++)
	{
		reduce(communicator(pg),localMaxHeight, maxHeightOfLeaf, boost::parallel::maximum<int>(),j);
		if(localMaxHeight < maxHeightOfLeaf)
			localMaxHeight = maxHeightOfLeaf;
	}
	maxHeightOfLeaf = localMaxHeight;
	if(procRank == 0)
	{
		printf("Total PRootLeaves:%ld \n",totalPRootLeaves);
	}
	long int *localLeavesAtHt = new long int[maxHeightOfLeaf+1];
	long int *leavesAtHt = new long int[maxHeightOfLeaf+1];
	//long in t totalLeaves=0;
	for(int i=0;i<maxHeightOfLeaf+1;i++)
	{
		localLeavesAtHt[i]=0;
		leavesAtHt[i]=0;
	}
	for (it=numLeavesAtHeight.begin(); it!=numLeavesAtHeight.end(); ++it)
		localLeavesAtHt[it->first] = it->second;
	for(int i=0;i<maxHeightOfLeaf+1;i++)
	{
		reduce(communicator(pg),localLeavesAtHt[i], leavesAtHt[i], std::plus<long int>(),0);
		if((procRank == 0) && (leavesAtHt[i] > 0))
		{
			printf("Total Leaves At Height: %d: %ld\n",i,leavesAtHt[i]);
			//totalLeaves += leavesAtHt[i];	
		}
	}
	delete [] leavesAtHt;
	delete [] localLeavesAtHt;
	numLeavesAtHeight.clear();
#endif	

	pLeafMap.erase(pLeafMap.begin(),pLeafMap.end());
#endif

	
return 0;
}

int GAL::GAL_BuildSubTree(TIndices& pointRefs, GAL_Vertex* subtreeRoot, int height, int DOR, bool isLeft, const Box& boundingBox)
{
	//If the subtree of configured height is created, ask a different process to take over the creation of subtrees that are down the hierarchy
	{
		//reached int node 
		int flag =BUILDSUBTREE_SAMEPROCESS;
		int leftFlag =BUILDSUBTREE_SAMEPROCESS;
		int rightFlag =BUILDSUBTREE_SAMEPROCESS;
		TIndices leftPoints, rightPoints;
		Box leftBox, rightBox;

		GAL_Vertex* intNode; 
		if((DOR == 0) || ((DOR % (subtreeHeight+1))==0))
		{
			intNode = GAL_CreateVertex(VTYPE_PSEUDOROOT);
			intNode->pseudoRoot = true;
		}
		else	
			intNode = GAL_CreateVertex(VTYPE_NORMAL);

		intNode->box = boundingBox;
		intNode->box.calcDimensions();
		//intNode->label = labelCount++;
		intNode->level = height;
		intNode->isLeftChild = isLeft;
		
		if(subtreeRoot != NULL)
		{
			//set parent descriptor
			intNode->parentDesc = subtreeRoot->desc.owner;
			//set parent pointer
			if((subtreeRoot->desc).owner == (intNode->desc).owner)
			{
				intNode->parent = subtreeRoot;
			}
			else
			{
				intNode->parent = subtreeRoot->parent;
			}
			
			//update parents left and right children pointers and descriptors and bounds
			if(isLeft)
			{
				subtreeRoot->leftDesc = procRank;
				subtreeRoot->leftChild = intNode;
			}
			else
			{
				subtreeRoot->rightDesc = procRank;
				subtreeRoot->rightChild = intNode;
			}
		}
		else
		{
			intNode->pseudoRoot = false;
			rootNode = intNode;
			rootNode->parent = NULL;
#ifdef TRAVERSAL_PROFILE
			pipelineStage.push_back(reinterpret_cast<long int>(rootNode));
#endif

		}
		
		if(pointRefs.size() > 64)	
		{
			bool ret = MedianSplit(intNode, height, pointRefs, leftPoints, rightPoints, leftBox, rightBox);
			if(!ret)
			{
#ifdef TRAVERSAL_PROFILE
#ifdef MERGE_HEIGHT1_TREES
				/*if(intNode->pseudoRoot)
				{
					long int deletedVertex= reinterpret_cast<long int>(intNode);
					std::vector<long int>::iterator iter = pipelineStage.begin();
					for(;iter!=pipelineStage.end();iter++)
						if(*iter==deletedVertex)
							break;
					pipelineStage.erase(iter);
				}*/
#endif
#endif
				return BUILDSUBTREE_SAMEPROCESS;
			}
		}
		else if(pointRefs.size() > 32)
		{
			bool ret = SAHSplit(intNode,height,pointRefs, leftPoints,rightPoints,leftBox,rightBox);
			if(!ret)
			{
#ifdef TRAVERSAL_PROFILE
#ifdef MERGE_HEIGHT1_TREES
				/*if(intNode->pseudoRoot)
				{
					long int deletedVertex= reinterpret_cast<long int>(intNode);
					std::vector<long int>::iterator iter = pipelineStage.begin();
					for(;iter!=pipelineStage.end();iter++)
						if(*iter==deletedVertex)
							break;
					pipelineStage.erase(iter);
				}*/
#endif
#endif
				return BUILDSUBTREE_SAMEPROCESS;
			}
		}	
		else
		{
			//intNode is leaf. Stop recursing.
			TriangleVector::iterator piter = points->begin();
			for(int i=0;i<pointRefs.size();i++)
				(intNode->triangles).push_back(*(piter+pointRefs[i]));
			pointRefs.erase(pointRefs.begin(),pointRefs.end());
#ifdef TRAVERSAL_PROFILE
			std::pair<std::map<int,long int>::iterator,bool> ret;
			ret = numLeavesAtHeight.insert(std::pair<int,long int>(height,1) );
			if(ret.second==false)
				numLeavesAtHeight[height] += 1;
#endif
#ifdef TRAVERSAL_PROFILE
#ifdef MERGE_HEIGHT1_TREES
				/*if(intNode->pseudoRoot)
				{
					long int deletedVertex= reinterpret_cast<long int>(intNode);
					std::vector<long int>::iterator iter = pipelineStage.begin();
					for(;iter!=pipelineStage.end();iter++)
						if(*iter==deletedVertex)
							break;
					pipelineStage.erase(iter);
				}*/
#endif
#endif
			return BUILDSUBTREE_SAMEPROCESS;
		}

		if(leftPoints.size() == 0)
			intNode->uCount++;
		if(rightPoints.size() == 0)
			intNode->uCount++;

		if((numProcs > 1) && (DOR == subtreeHeight))
		{
			long int pLeaf = reinterpret_cast<long int>(intNode);
			int nextprocessid = GAL_GetNextProcessId(subtreeHeight, pLeaf);
#ifdef MERGE_HEIGHT1_TREES
			if(leftPoints.size() > 32)
#else
			if(leftPoints.size() > 0)
#endif
			{
#ifdef LOAD_BALANCE
				pseudoLeaves.push_back(intNode);
				if((procRank != 0) && (subtreeHeight == intNode->level))
				{
					leftFlag = BUILDSUBTREE_MAXHEIGHT;
					flag = BUILDSUBTREE_MAXHEIGHT;
				}
				else
#endif
				{
				intNode->pseudoLeaf = true;
				MsgBuildSubTree m(true,height+1,intNode->desc);
				m.pLeaf = pLeaf;//reinterpret_cast<long int>(intNode);
				m.ptv = leftPoints;
				m.box=leftBox;
				m.pLeafLabel = intNode->id;
				GAL_UpdateLocalNodeCount(nextprocessid,1);
				GAL_SendMessage(nextprocessid,MESSAGE_BUILDSUBTREE,&m); 
				leftFlag = BUILDSUBTREE_MAXHEIGHT;
				flag = BUILDSUBTREE_MAXHEIGHT;
				}
			}
#ifdef MERGE_HEIGHT1_TREES
			if(rightPoints.size() > 32)
#else
			if(rightPoints.size() > 0)
#endif
			{
#ifdef LOAD_BALANCE
				if(leftFlag != BUILDSUBTREE_MAXHEIGHT)
				{
					pseudoLeaves.push_back(intNode);
				}
				if((procRank != 0) && (subtreeHeight == intNode->level))
				{
					rightFlag = BUILDSUBTREE_MAXHEIGHT;
					flag = BUILDSUBTREE_MAXHEIGHT;
				}
				else
#endif
				{
				intNode->pseudoLeaf = true;
				MsgBuildSubTree m(false,height+1,intNode->desc);
				m.pLeaf = pLeaf;//reinterpret_cast<long int>(intNode);
				m.ptv = rightPoints;
				m.box=rightBox;
				m.pLeafLabel = intNode->id;
				GAL_UpdateLocalNodeCount(nextprocessid,1);
				GAL_SendMessage(nextprocessid,MESSAGE_BUILDSUBTREE,&m); 
				rightFlag = BUILDSUBTREE_MAXHEIGHT;
				flag = BUILDSUBTREE_MAXHEIGHT;
				}
			}
		}
		
		/*if((leftPoints.size() > 0) && (rightPoints.size() > 0))
		//printf("current_depth:%d leftPoints:%d rightPoints:%d selfpoints:0 leftBox min=(%f %f %f) leftBox max=(%f %f %f) rightBox min=(%f %f %f) rightBox max=(%f %f %f)\n",height,leftPoints.size(),rightPoints.size(), leftBox.min[0],leftBox.min[1],leftBox.min[2],leftBox.max[0],leftBox.max[1],leftBox.max[2],rightBox.min[0],rightBox.min[1],rightBox.min[2],rightBox.max[0],rightBox.max[1],rightBox.max[2]);
		printf("%d: current_depth:%d leftPoints:%d rightPoints:%d \n",procRank,height,leftPoints.size(),rightPoints.size());
		else if((leftPoints.size() > 0))
		//printf("current_depth:%d leftPoints:%d rightPoints:%d selfpoints:0 leftBox min=(%f %f %f) leftBox max=(%f %f %f) rightBox min=(0.0 0.0 0.0) rightBox max=(0.0 0.0 0.0)\n",height,leftPoints.size(),rightPoints.size(), leftBox.min[0],leftBox.min[1],leftBox.min[2],leftBox.max[0],leftBox.max[1],leftBox.max[2]);
		printf("%d: current_depth:%d leftPoints:%d rightPoints:%d \n",procRank,height,leftPoints.size(),rightPoints.size());
		else if((rightPoints.size() > 0))
		//printf("current_depth:%d leftPoints:%d rightPoints:%d selfpoints:0 leftBox min=(0.0 0.0 0.0) leftBox max=(0.0 0.0 0.0) rightBox min=(%f %f %f) rightBox max=(%f %f %f)\n",height,leftPoints.size(),rightPoints.size(), rightBox.min[0],rightBox.min[1],rightBox.min[2],rightBox.max[0],rightBox.max[1],rightBox.max[2]);
		printf("%d: current_depth:%d leftPoints:%d rightPoints:%d \n",procRank,height,leftPoints.size(),rightPoints.size());
		//printf("current_depth:%d leftPoints:%d rightPoints:%d selfpoints:0\n",height,leftPoints.size(),rightPoints.size());*/
		//build left tree
		if((leftPoints.size() > 0) && (leftFlag != BUILDSUBTREE_MAXHEIGHT))
		{
			int ret = GAL_BuildSubTree(leftPoints,intNode,height+1, DOR+1, true, leftBox);
			if(ret == BUILDSUBTREE_SAMEPROCESS)
				intNode->uCount++;
			else
				flag = ret;
		}

		//build right tree
		if((rightPoints.size() >0) && (rightFlag != BUILDSUBTREE_MAXHEIGHT))	
		{
			int ret  = GAL_BuildSubTree(rightPoints,intNode, height+1, DOR+1, false, rightBox);
			if(ret == BUILDSUBTREE_SAMEPROCESS)
				intNode->uCount++;
			else
				flag = ret;
		}
		

		if(flag == BUILDSUBTREE_SAMEPROCESS)
		{
			assert(intNode->uCount == 2);
			if(!(intNode->leftChild) && !(intNode->rightChild))
			{
#ifdef TRAVERSAL_PROFILE
				std::pair<std::map<int,long int>::iterator,bool> ret;
				ret = numLeavesAtHeight.insert(std::pair<int,long int>(height,1) );
				if(ret.second==false)
					numLeavesAtHeight[height] += 1;
#endif

				TriangleVector::iterator piter = points->begin();
				for(int i=0;i<pointRefs.size();i++)
					(intNode->triangles).push_back(*(piter+pointRefs[i]));
			}
		}
		pointRefs.erase(pointRefs.begin(),pointRefs.end());

		return flag;

	}
	
}

void GAL::GAL_PrintGraph()
{

#ifdef TRAVERSAL_PROFILE
	long int totalPointsVisited =0;
	//to find the bottlenecks uncomment the below line and all its occurences.
	std::vector<std::pair<long int, long int> > footprintMatrix;
	std::vector<long int>::iterator iter = pipelineStage.begin();
	for(;iter!=pipelineStage.end();iter++)
	{
		GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(*iter);
		if((pRoot->level != 0))// && (pRoot->leftChild || pRoot->rightChild))
		{
			long int count=0;
			GALHelper_CountSubtreeNodes(pRoot, count, true);
			//printf("%d Subtree %ld Points_Visited %ld\n",procRank, pRoot->id,count);
			footprintMatrix.push_back(std::make_pair(*iter,count));
			totalPointsVisited += count;
		}
		else if(pRoot->level == 0)
		{
			long int count=0;
			GALHelper_CountSubtreeNodes(pRoot, count, true, true);
			//printf("%d Subtree %ld Points_Visited %ld\n",procRank, pRoot->id,count);
			totalPointsVisited += count;
		}

		/*if(pRoot->level == 0)
		{
			long int count=0;
			GALHelper_CountSubtreeNodes(pRoot, count);
			printf("Subtree %ld Points_Visited %ld\n",*iter,count);
			double bof=0.;
			GALHelper_GetBlockOccupancyFactor(pRoot, bof);
			printf("Bof:  %f\n",bof/count);
		}*/
	}
	//printf("%d: number of subtrees:%d average number of visitors:%f\n",procRank,pipelineStage.size(),totalPointsVisited/(float)pipelineStage.size());
	//Finding bottlenecks.
	if(footprintMatrix.size() > 0)
		std::sort(footprintMatrix.begin(),footprintMatrix.end(),SortLongIntPairsBySecond_Decrease);
	int numberOfBottlenecks=7758,j=0;
	while(j < numberOfBottlenecks)
	{
		int i=0;
		long int localElem=-1, globalElem=-1;
		if(i<footprintMatrix.size())
			localElem = footprintMatrix[i].second;	
		all_reduce(communicator(pg),localElem,globalElem,boost::parallel::maximum<long int>());
		if(i<footprintMatrix.size())
		{
			//printf("localElem:%d globalElem:%d\n",localElem,globalElem);
			if(localElem == globalElem)
			{
				GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(footprintMatrix[i].first);
				int childNum = pRoot->isLeftChild?0:1;
				footprintMatrix.erase(footprintMatrix.begin());
				printf("%d %ld %d %d\n",procRank,((GAL_PseudoRoot*)pRoot)->pLeafLabel, pRoot->id,childNum);
			}	
		}	
		j++;
	}

#endif

#ifdef STATISTICS
	long int totalTraverseCount=0;
	reduce(communicator(pg),traverseCount, totalTraverseCount, std::plus<long int>(),0);
	float localPPM=0., avgPPM;
	if(pointsProcessed != 0)
		localPPM = pointsProcessed/(float)traverseCount;
		
	reduce(communicator(pg),localPPM, avgPPM, std::plus<float>(),0);
	if(procRank==0)
		printf("Total Messages Processed:%ld Avergage Points Per Message:%f\n",totalTraverseCount, avgPPM/numProcs);


#ifdef MESSAGE_AGGREGATION
	if(bufferStage)
	{
		long int totalBufferCount=0;
		for(int i=1;i<(numPipelineBuffers+1);i++)
		{
			totalBufferCount=0;
			reduce(communicator(pg),bufferStage[i], totalBufferCount, std::plus<long int>(),0);
			if(procRank == 0)
				printf("Total Points Buffered at stage %d: %ld\n",i,totalBufferCount);
		}
		delete [] bufferStage;
		bufferStage = NULL;
	}
#endif

#endif


}

int GAL::GALHelper_HandleMessageTraverseBackward(GAL_Vertex* childNode, GALVisitor* vis, TBlockId curBlockId)
{
	int ret = STATUS_TRAVERSE_INCOMPLETE;
	GAL_Vertex* tmpChildNode = childNode;
	GAL_Vertex* nextNodeToVisit = NULL;
#ifdef PERFCOUNTERS
	assert(compTimer.IsRunning());
#endif

	while(true)
	{
#ifdef MESSAGE_AGGREGATION
		TBlockId fragBlkId;
		bool readyToSend=false;	
		if(tmpChildNode->pseudoRoot)
		{
			fragBlkId.second = INVALID_BLOCK_ID; 
			bool readyToSend = vis->GALVisitor_IsLastFragment(tmpChildNode, reinterpret_cast<BlockStack*>(curBlockId.second), fragBlkId);	
			if(fragBlkId.second != INVALID_BLOCK_ID)
			{
				vis->GALVisitor_SetAsCurrentBlockStack2(reinterpret_cast<BlockStack*>(curBlockId.second));
				vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
				vis->GALVisitor_RemoveBlockStack();
				if(!readyToSend)
				{
					break;
				}
				else
				{	
					curBlockId = fragBlkId;
					vis->GALVisitor_SetAsCurrentBlockStack(curBlockId);
				}
			}
		}
#endif
		long int parent = reinterpret_cast<long int>(tmpChildNode->parent);
		int procId = tmpChildNode->parentDesc;

#ifdef LOAD_BALANCE
		if(tmpChildNode->pseudoRoot && (tmpChildNode->level == (subtreeHeight+1)))
		{
			BlockStack* curBStack = reinterpret_cast<BlockStack*>(curBlockId.second);
			procId = curBStack->parentBlockId.first;
			assert(procId < numProcs);
			std::vector<RepPLeaf>::iterator pLeafIter=(((GAL_PseudoRoot*)(tmpChildNode))->parents).begin();
			for(;pLeafIter!= (((GAL_PseudoRoot*)(tmpChildNode))->parents).end();pLeafIter++)
			{
				if((pLeafIter->first) == procId)
				{
					parent = pLeafIter->second;
					break;
				}
			}
			assert(parent != 0);
			if(procId == 0)
				parent = reinterpret_cast<long int>(tmpChildNode->parent);
		}
#endif
		/*if(curBlockId.second == 67110524)
		{
			printf("Debug 8\n");
		}*/
		nextNodeToVisit  = vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();

		if(tmpChildNode->pseudoRoot && (procId != procRank))
		{
			MsgTraverse msgTraverse;
			GAL_PseudoRoot* pRoot=(GAL_PseudoRoot*)tmpChildNode;
			GAL_Vertex* tmpNextNodeToVisit = NULL;
			int nextNodeProc=0;
			TBlockId blkStart = vis->GALVisitor_DeleteFromSuperBlock(pRoot, &tmpNextNodeToVisit, nextNodeProc);	
			vis->GALVisitor_GetLocalData(msgTraverse.l);
			msgTraverse.blkStart = blkStart;
			msgTraverse.pLeaf = parent;//reinterpret_cast<long int>(tmpChildNode->parent);
			msgTraverse.pSibling = static_cast<long int>(0);
			if(!nextNodeToVisit)
			{
#ifdef PERFCOUNTERS
					compTimer.Stop();
#endif
				//vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
				int re = vis->GALVisitor_RemoveBlockStack();
				msgTraverse.pRoot = reinterpret_cast<long int>(tmpChildNode);
				GAL_SendMessage(procId,MESSAGE_TRAVERSE_BACKWARD, &msgTraverse);
				ret = STATUS_TRAVERSE_INCOMPLETE;
			}
			else
			{
				//if((pRoot->siblingDesc).owner != procRank)
				if(nextNodeProc != procRank)
				{
#ifdef PERFCOUNTERS
					compTimer.Stop();
#endif
					//vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
					if(procId != procRank)
					int re = vis->GALVisitor_RemoveBlockStack();		
					//sending entire compressed data to sibling
					msgTraverse.pRoot = reinterpret_cast<long int>(nextNodeToVisit);
					msgTraverse.pSibling = static_cast<long int>(0);
					GAL_SendMessage(nextNodeProc,MESSAGE_TRAVERSE, &msgTraverse);
					ret = STATUS_TRAVERSE_INCOMPLETE;
					break;
				}
				else
				{
#ifdef MESSAGE_AGGREGATION
					vis->GALVisitor_RemoveBlockStack();
					vis->GALVisitor_PushToBlockStackAndUpdate(msgTraverse);
#endif
					ret = GALHelper_DeAggregateBlockAndSend(vis, &nextNodeToVisit, msgTraverse, false);
					if(ret == STATUS_TRAVERSE_COMPLETE)
					{
						msgTraverse.l.clear();
						vis->GALVisitor_GetLocalData(msgTraverse.l);
						int re = vis->GALVisitor_RemoveBlockStack();
						msgTraverse.pRoot = reinterpret_cast<long int>(nextNodeToVisit);
						GAL_SendMessage(procId,MESSAGE_TRAVERSE_BACKWARD, &msgTraverse);
						ret = STATUS_TRAVERSE_INCOMPLETE;
					}
				}	
			}
			return ret;
		}
		else if(tmpChildNode->pseudoRoot && (procId == procRank))
		{
			GAL_Vertex* prnt = reinterpret_cast<GAL_Vertex*>(parent);//tmpChildNode->parent;
			if(prnt == 0)
			{
				ret = 1;	
				break;
			}
			if(nextNodeToVisit)
			{
				//load balanced. traversing backedge.
				int nextNodeProc = 0;
				if(tmpChildNode->isLeftChild)
				{
					nextNodeProc = prnt->rightDesc;
				}
				else
				{
					nextNodeProc = 	prnt->leftDesc;
				}
				MsgTraverse msgTraverse;
				msgTraverse.pRoot = reinterpret_cast<long int>(nextNodeToVisit);
				msgTraverse.pLeaf = parent;//reinterpret_cast<long int>(tmpChildNode->parent);
				msgTraverse.pSibling = static_cast<long int>(0);

				if(nextNodeProc != procRank)
				{
#ifdef PERFCOUNTERS
					compTimer.Stop();
#endif
					TBlockId blkStart = vis->GALVisitor_GetLocalData(msgTraverse.l);
					msgTraverse.blkStart = blkStart;
					GAL_SendMessage(nextNodeProc,MESSAGE_TRAVERSE, &msgTraverse);
					ret = STATUS_TRAVERSE_INCOMPLETE;
					break;
				}
				else
				{
					ret = GALHelper_DeAggregateBlockAndSend(vis, &nextNodeToVisit, msgTraverse, true);
					if(ret != STATUS_TRAVERSE_COMPLETE)
						break;
					nextNodeToVisit = NULL;
				}
			}
		}
		

		if(nextNodeToVisit != NULL)
		{
			busyTime.Start();
			ret = GAL_TraverseHelper(vis,nextNodeToVisit, NULL);
			busyTime.Stop();
			if(ret != STATUS_TRAVERSE_COMPLETE)
			{
				break;
			}
			//parent of tmpChildNode and NextNode are same.
		}

		if(tmpChildNode->parent == NULL) 
		{
			ret = 1;
			break;
		}
		
		tmpChildNode = reinterpret_cast<GAL_Vertex*>(parent);//tmpChildNode->parent;
	}
	return ret;	
	
}



int GAL::GAL_TraverseHelper_SendBlocks(GALVisitor* vis)
{
	int ret =STATUS_TRAVERSE_INCOMPLETE;
	int status = STATUS_SUCCESS;

#ifdef PERFCOUNTERS
	std::map<long int, VisitedBlocks>::iterator iter;
	bool found = false;
	long int origBlkId;
	assert(!compTimer.IsRunning());
	compTimer.Start();
#endif
	
	assert(vis);
#ifndef LOAD_BALANCE
	if(procRank == 0)
#endif
	{
		do
		{
			status = vis->GALVisitor_FillPipeline(rootNode);
			if((status == STATUS_NO_WORKITEMS) ||(status == STATUS_PIPELINE_FULL)) 
			{
				if((status == STATUS_NO_WORKITEMS) && (vis->GALVisitor_GetNumberOfWorkItems() == 0))
					ret=STATUS_TRAVERSE_COMPLETE;
				else
				{
					ret=STATUS_TRAVERSE_INCOMPLETE;
#ifdef MESSAGE_AGGREGATION
					if(!readyToFlush)
					{
						numReadyToFlushUpdates++;
						//if(numReadyToFlushUpdates == numProcs)
						{
							readyToFlush = true;
						}
#ifdef PERFCOUNTERS
						compTimer.Stop();
#endif
						/*for(int i=0;i<numProcs;i++)
						{
							int sentVal = i;
							if(i != procRank)
								GAL_SendMessage(i,MESSAGE_SENDCOMPRESSED,&sentVal); 
						}*/
					}

					/*if(pipelineBufferTimerCount > 0)
						pipelineBufferTimerCount--;*/
#endif

				}

			}
			else
			{
				busyTime.Start();
				int ret = GAL_TraverseHelper(vis,rootNode, NULL);
				busyTime.Stop();
				
				if(ret == STATUS_TRAVERSE_INCOMPLETE)
				{
					status = STATUS_FAILURE;
				}
				else if(ret == STATUS_TRAVERSE_COMPLETE)
				{
					int re = vis->GALVisitor_RemoveBlockStack();		
				}

			}
		}
		while(status == STATUS_SUCCESS);
	}

#ifdef PERFCOUNTERS
	if(compTimer.IsRunning())
		compTimer.Stop();
	uint64_t stageExTime = compTimer.GetLastExecTime();
	if(found)
	{
		std::map<long int, PipelineStats>::iterator blkIter = (iter->second).find(origBlkId);
		assert(blkIter != (iter->second).end());
		(blkIter->second).timeInStage += stageExTime;
	}
#endif

#ifdef MESSAGE_AGGREGATION
	if(readyToFlush)
	//if(pipelineBufferTimerCount == 0)
	{
		if(aggrBuffer.size() > 0)
		{
			GAL_TraverseHelper_SendCompressedMessages(vis);
		}
	}
#endif

	return ret;
}





int GAL::GAL_Traverse(GALVisitor* vis)
{
	//dbg++;
	PBGL_oobMessage msg;
	bool done = false;
	int ret = STATUS_TRAVERSE_INCOMPLETE;
	double startTime, endTime;
#ifdef PERFCOUNTERS
	totTimer.Start();
	compTimer.Start();
	workloadTimer.Start();
#endif	
#ifdef MESSAGE_AGGREGATION
	assert((numPipelineBuffers > 0) && (numPipelineBuffers <= 3));
#ifdef STATISTICS
	if(!bufferStage)
	{
		bufferStage = new long int[numPipelineBuffers+1];
		for(int i=0;i<(numPipelineBuffers+1);i++)
			bufferStage[i]=0;	
	}
#endif
#endif
	
	startTime = clock();
#ifdef PERFCOUNTERS
	double lastTime = clock();
	compTimer.Stop();
#endif
	while (!done)
	{
		//poll for messages
		//usleep(100);
#ifdef PERFCOUNTERS
		if((clock() - lastTime) >= timeStep)
		{
			BlockStats bs = vis->GALVisitor_GetNumberOfBlocksInFlight();
			tpFinishedBlocks.push_back(bs.numUniqBlocks);
			lastTime = clock();
		}
#endif
		PBGL_AsyncMsg pollMsg = pg.poll();
		if(!pollMsg)
		{
			if(vis)
			{
				/*		if(dbg == 2)
							printf("%d: phase 2 sending blocks\n",procRank);*/
				ret = GAL_TraverseHelper_SendBlocks(vis);
				if(ret == STATUS_TRAVERSE_COMPLETE)
				{
#ifdef LOAD_BALANCE
					if(!readyToExit)
					{
#ifdef PERFCOUNTERS
						workloadTimer.Stop();
#endif

						readyToExitList.insert(procRank);
						if(readyToExitList.size() != numProcs)
						{
							for(int i=0;i<numProcs;i++)
							{
								if(i !=procRank)
									GAL_SendMessage(i,MESSAGE_READYTOEXIT,&procRank); 
							}
							readyToExit = true;
						}
						else
						{
							MsgTerminate msgTerminate;
							msgTerminate.pLeft = ret;
							for(int i=0;i<numProcs;i++)
							{
								if(i !=procRank)
									GAL_SendMessage(i,MESSAGE_DONEKDTREE,&msgTerminate); 
							}
							break;
						}
					}
#else
					MsgTerminate msgTerminate;
					msgTerminate.pLeft = ret;
					for(int i=0;i<numProcs;i++)
					{
						if(i !=procRank)
							GAL_SendMessage(i,MESSAGE_DONEKDTREE,&msgTerminate); 
					}
					break;
#endif
				}
			}
			continue;
		}
		else
		{
			msg = pollMsg.get();

						/*if(dbg == 2)
							printf("%d: phase 2 receving message. sender:%d msgId:%d\n",procRank,msg.first, msg.second);*/
			//poll for messages
			switch(msg.second)
			{
				case MESSAGE_TRAVERSE:
				{
						MsgTraverse msgTraverse;
						//receive_oob(pg, msg->first, msg->second, msgTraverse);
						receive_oob(pg, msg.first, msg.second, msgTraverse);
						GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(msgTraverse.pRoot);
						GAL_PseudoRoot* pSibling = reinterpret_cast<GAL_PseudoRoot*>(msgTraverse.pSibling);
#ifdef STATISTICS
						traverseCount++;
						pointsProcessed += msgTraverse.l.size();
#endif
						int procId =(pRoot->parentDesc);
						bool loadBalanced = false;
						if(pRoot->level == subtreeHeight+1)
						{
							procId = msgTraverse.blkStart.first;
							if(procId == procRank)
							{
								loadBalanced = true;
							}
						}
						int re = 0;
						if(!loadBalanced)
							vis->GALVisitor_PushToBlockStackAndUpdate(msgTraverse);
#ifdef MESSAGE_AGGREGATION
						else
						{
							if(vis->GALVisitor_IsBufferedBlock(msgTraverse.blkStart.second))
							{
								vis->GALVisitor_SetLocalData(msgTraverse.l, msgTraverse.blkStart, false);
								TCBSet blkIdSet;
								int status = vis->GALVisitor_GetCompressedBlockIDs(msgTraverse.blkStart,msgTraverse.l[0],blkIdSet);
								if(status == STATUS_SUCCESS)
								{
									std::vector<RepPLeaf>::iterator pLeafIter=(((GAL_PseudoRoot*)(pRoot))->parents).begin();
									long int parent = reinterpret_cast<long int>(pRoot->parent);
									for(;pLeafIter!= (((GAL_PseudoRoot*)(pRoot))->parents).end();pLeafIter++)
									{
										if(procRank == 0)
											break;
										if((pLeafIter->first) == procRank)
										{
											parent = pLeafIter->second;
											break;
										}
									}

									TCBSet::iterator sIter = blkIdSet.begin();
									for(;sIter!=blkIdSet.end();sIter++)
									{
										vis->GALVisitor_SetAsCurrentBlockStack2(*sIter);
										TBlockId curBlockId = (*sIter)->bStackId;
										msgTraverse.l.clear();
										vis->GALVisitor_GetLocalData(msgTraverse.l);
										msgTraverse.blkStart =curBlockId;
										ret = GALHelper_DeAggregateBlockAndSend(vis, &pRoot, msgTraverse, false);
										if(ret == STATUS_TRAVERSE_COMPLETE)
										{
											status = GALHelper_HandleMessageTraverseBackward(reinterpret_cast<GAL_Vertex*>(parent),vis,curBlockId);
											if(status == 1)
											{
												int re = vis->GALVisitor_RemoveBlockStack();		
											}
										}
									}
								}
								break;
							}
						}
#endif

						if(loadBalanced)
							re = vis->GALVisitor_SetLocalData(msgTraverse.l,msgTraverse.blkStart);

						if(re == -1)
							printf("from:%d FORWARD pRootLevel:%d procId:%d\n",msg.first,pRoot->level, procId);	
						
						ret = GALHelper_DeAggregateBlockAndSend(vis, &pRoot, msgTraverse, loadBalanced);

						if(ret == STATUS_TRAVERSE_COMPLETE)
						{
							if(!loadBalanced)
							{
								int re = vis->GALVisitor_RemoveBlockStack();	
								if(re == -1)
									printf("Debug 6\n");
							}
							if(procId != procRank)	
							{
#ifdef PERFCOUNTERS
								compTimer.Stop();
#endif
								GAL_SendMessage(procId,MESSAGE_TRAVERSE_BACKWARD, &msgTraverse);
								ret = STATUS_TRAVERSE_INCOMPLETE;
							}
							else
							{
								bool decodeBlocks = false;
#ifdef MESSAGE_AGGREGATION
								TCBSet blkIdSet;
								int stageNum = PIPELINE_STAGE_NUM(pRoot->parent->level, numPipelineBuffers, subtreeHeight);
								if(stageNum <= numPipelineBuffers)
								{
									int status = vis->GALVisitor_GetCompressedBlockIDs(msgTraverse.blkStart,msgTraverse.l[0],blkIdSet);
									if(status == STATUS_SUCCESS)
									{
										decodeBlocks = true;
										status = GALHelper_HandleMessageTraverseBackward_Multiple(pRoot->parent, vis, blkIdSet);
									}
								}
#endif
								if(!decodeBlocks)
								{
									vis->GALVisitor_SetAsCurrentBlockStack(msgTraverse.blkStart);
									GAL_Vertex* pLeaf = reinterpret_cast<GAL_Vertex*>(msgTraverse.pLeaf);
									int status = GALHelper_HandleMessageTraverseBackward(pLeaf, vis, msgTraverse.blkStart);
									if(status == 1)
									{
										int re = vis->GALVisitor_RemoveBlockStack();		
										if(re == -1)
											printf("Debug 7\n");
									}
								}
							}
						}
#ifdef PERFCOUNTERS
						if(leafStageTimer.IsRunning())
							leafStageTimer.Stop();
						if(compTimer.IsRunning())
							compTimer.Stop();
						uint64_t stageExTime = compTimer.GetLastExecTime();
						if(found)
						{
							if(!(doesBlockExist.second))
							{
								((doesBlockExist.first)->second).timeInStage += stageExTime;
							}
							else
							{
							}
						}
#endif
							
				}
				break;
				case MESSAGE_TRAVERSE_BACKWARD:
				{
						MsgTraverse msgBkTraverse;
						int status;
						receive_oob(pg, msg.first, msg.second, msgBkTraverse);
						GAL_Vertex* parentNode = reinterpret_cast<GAL_Vertex*>(msgBkTraverse.pLeaf);
						GAL_Vertex* childNode = reinterpret_cast<GAL_Vertex*>(msgBkTraverse.pRoot);
						bool decodeBlocks = false;
						//traverseCount++;
#ifdef PERFCOUNTERS
						compTimer.Start();
#endif
						/*if(msgBkTraverse.pLeaf == 0xce1c90)//(msgBkTraverse.blkStart.second == 501))
						{
							printf("Debug 8\n");
						}*/
#ifdef MESSAGE_AGGREGATION
						TCBSet blkIdSet;
						int stageNum = PIPELINE_STAGE_NUM(parentNode->level, numPipelineBuffers, subtreeHeight);
						if(stageNum <= numPipelineBuffers)
						{
							int status = vis->GALVisitor_GetCompressedBlockIDs(msgBkTraverse.blkStart,msgBkTraverse.l[0],blkIdSet);
							if(status == STATUS_SUCCESS)
							{
								decodeBlocks = true;
								vis->GALVisitor_SetLocalData(msgBkTraverse.l, msgBkTraverse.blkStart, false);
								GALHelper_HandleMessageTraverseBackward_Multiple(parentNode, vis, blkIdSet);
							}
							else
							{	
								//could be aggregate or non-aggregated block from upper subtree.	
								//so treat it as a monolithic block. Do nothing.
							}
						}

						if(!decodeBlocks)
#endif
						{
							bool waitForSibling = false;
							/*if((dbg == 9) && (msgBkTraverse.blkStart.second == 3))
								printf("break\n");*/
							int re = vis->GALVisitor_SetLocalData(msgBkTraverse.l, msgBkTraverse.blkStart);
							status = GALHelper_HandleMessageTraverseBackward(parentNode, vis, msgBkTraverse.blkStart);
							if(status == 1)
							{
									int re = vis->GALVisitor_RemoveBlockStack();		
							}
						}
#ifdef PERFCOUNTERS
						if(compTimer.IsRunning())
							compTimer.Stop();
						uint64_t stageExTime = compTimer.GetLastExecTime();
						if(found)
						{
							if(!(doesBlockExist.second))
							{
								((doesBlockExist.first)->second).timeInStage += stageExTime;
							}
							else
							{
								std::map<long int, PipelineStats>::iterator blkIter = (iter->second).find(msgBkTraverse.blkStart.second);
								assert(blkIter != (iter->second).end());
								(blkIter->second).timeInStage += stageExTime;
							}
						}

#endif
						
				}
				break;
#ifdef MESSAGE_AGGREGATION
				case MESSAGE_SENDCOMPRESSED:
				{
					int recvdVal;
					receive_oob(pg, msg.first, msg.second, recvdVal);
					numReadyToFlushUpdates++;
					if(numReadyToFlushUpdates == numProcs)
					{
						readyToFlush = true;
					}
				}
				break;
#endif

				case MESSAGE_DONEKDTREE:
				{
					MsgTerminate msgTerminate;
					receive_oob(pg, msg.first, msg.second, msgTerminate);
					ret = msgTerminate.pLeft;
					done = true;
				}
				break;
#ifdef LOAD_BALANCE
				case MESSAGE_READYTOEXIT:
				{
					int doneProcId;
					receive_oob(pg, msg.first, msg.second, doneProcId);
					readyToExitList.insert(doneProcId);
					if(readyToExitList.size() == numProcs)
					{
						done = true;
						/*if(procRank == 0)
						{
							for(int i=0;i<numProcs;i++)
							{
								if(i !=procRank)
									GAL_SendMessage(i,MESSAGE_DONEKDTREE,&doneProcId); 
							}
						}*/
					}
				}
				break;
#endif
				default: break;
			}
		}
	}
#ifdef LOAD_BALANCE
	readyToExitList.clear();
	readyToExit=false;
#endif

#ifdef MESSAGE_AGGREGATION
	numReadyToFlushUpdates = 0;
	readyToFlush = false;
#endif
#ifdef PERFCOUNTERS
	workloadTime += workloadTimer.GetTotTime();
	totTimer.Stop();
	totTime += totTimer.GetTotTime();
	compTime += compTimer.GetTotTime();
	totTimer.Reset();
	compTimer.Reset();
#endif	

	endTime = clock();
	traversalTime += (endTime-startTime)/CLOCKS_PER_SEC;
	
	
	return ret;
}

int GAL::GAL_TraverseHelper(GALVisitor* vis, GAL_Vertex* node, GAL_Vertex* sib)
{

		if(node == NULL)
			return STATUS_TRAVERSE_COMPLETE;

		TIndices leftBlock, rightBlock, superBlock;
		int ret = STATUS_TRAVERSE_COMPLETE;
		TBlockId blockId;
		long int uniqBlkId1, uniqBlkId2;
		assert((node->desc).owner ==procRank); 
		BlockStackListIter rightBStack, curBStack = vis->GALVisitor_GetCurrentBlockStack();
		
		bool traverseSubTree = vis->GALVisitor_VisitNode(node, sib, leftBlock, rightBlock, blockId);
		if(!traverseSubTree)
		{
			return STATUS_TRAVERSE_COMPLETE;
		}


		
		
		//traverse left subtree
		if(leftBlock.size() > 0)
		{
			leftBlock.erase(leftBlock.begin(),leftBlock.end());
			if(node->leftChild && ((node->leftDesc) != procRank))
			{	
#ifdef PERFCOUNTERS
				compTimer.Stop();
#endif
				MsgTraverse msgTraverse, msgTraverseRight;
				blockId = vis->GALVisitor_GetLocalData(msgTraverse.l);
				msgTraverse.pRoot = reinterpret_cast<long int>(node->leftChild);
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
				int stageNum = PIPELINE_STAGE_NUM(node->level, numPipelineBuffers, subtreeHeight);
				int aggrBlockSize = DEFAULT_BUFFER_SIZE;
				int bufCount;

				/*switch(stageNum)
				{
					case 1: aggrBlockSize =  PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,1);
						break;
					case 2: aggrBlockSize = PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,2);
						break;
					case 3: aggrBlockSize = PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,3);
						break;
					default: aggrBlockSize = DEFAULT_BUFFER_SIZE;
						break;
				}*/
				bufCount = aggrBlockSize;
				if(stageNum <= numPipelineBuffers)
				{
				    aggrBlockSize = GetOpt_PipelineBufferSizes(stageNum -1);
					bufCount = GAL_TraverseHelper_CompressMessages(vis,msgTraverse, aggrBlockSize, msgTraverseRight,true);
				}

			if(bufCount >= aggrBlockSize)
			{
#endif
			GAL_SendMessage((node->leftDesc),MESSAGE_TRAVERSE, &msgTraverse);
			ret = STATUS_TRAVERSE_INCOMPLETE;
#ifdef MESSAGE_AGGREGATION
				if(msgTraverseRight.l.size() > 0)
				{
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
					GAL_SendMessage((node->rightDesc),MESSAGE_TRAVERSE, &msgTraverseRight);
					ret = STATUS_TRAVERSE_INCOMPLETE;
				}
		
			}
			else
			{
#ifdef MESSAGE_AGGREGATION
						if(pipelineBufferTimerCount == 0)
						{
							pipelineBufferTimerCount = PIPELINE_BUFFER_TIMERVAL;
						}
#endif
				ret = STATUS_TRAVERSE_INCOMPLETE;
			}
#ifdef STATISTICS
				if(bufCount>=aggrBlockSize)
				{
				//printf("node->level:%d, stagenum %d numPipelinebuffers:%d bufCount:%d aggrBlockSize:%d\n",node->level, stageNum, numPipelineBuffers, bufCount, aggrBlockSize);	
					if(stageNum <= numPipelineBuffers)
						bufferStage[stageNum] += bufCount;
				}
#endif
#endif
		}
		else
		{
			busyTime.Start();
			ret = GAL_TraverseHelper(vis,node->leftChild, node->rightChild);
			busyTime.Stop();
			if(ret == STATUS_TRAVERSE_COMPLETE)
			{
				if(node->rightChild && ((node->rightDesc) != procRank))
				{	
#ifdef PERFCOUNTERS
					compTimer.Stop();
#endif
					MsgTraverse msgTraverse;
					blockId = vis->GALVisitor_GetLocalData(msgTraverse.l);

					msgTraverse.blkStart = blockId;
					msgTraverse.pRoot = reinterpret_cast<long int>(node->rightChild);
					msgTraverse.pLeaf = reinterpret_cast<long int>(node);
					msgTraverse.pSibling = static_cast<long int>(0);
					msgTraverse.siblingDesc = node->rightDesc;
					GAL_SendMessage((node->rightDesc),MESSAGE_TRAVERSE, &msgTraverse);
					ret = STATUS_TRAVERSE_INCOMPLETE;
				}
				else
				{
					//node->rightDesc is ignored.
					busyTime.Start();
					ret = GAL_TraverseHelper(vis,node->rightChild, NULL);
					busyTime.Stop();
						if(ret == STATUS_TRAVERSE_COMPLETE)			
						{
							vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
						}	
					}
				}
					
			}
				
			
		}
		

		if(ret != STATUS_TRAVERSE_COMPLETE)		
		{
#ifdef PERFCOUNTERS
			if(compTimer.IsRunning())
				compTimer.Stop();
#endif
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
int GAL::GAL_TraverseHelper_CompressMessages(GALVisitor* vis,MsgTraverse& msgTraverse, int aggrBlockSize, MsgTraverse& msgTraverseRight, bool goingLeft)
{
	TBlockId blockId;
	TIndices tmpIndices;
	long int pseudoLeaf;
	if(goingLeft)
		pseudoLeaf = msgTraverse.pLeaf;
	else
		pseudoLeaf = msgTraverseRight.pLeaf;
		
	//pipelineBufferTimerCount=PIPELINE_BUFFER_TIMERVAL;
	BlockStack* bufferStack;
	int aggrSize=0;
	//Get the current buffer at the vertex
	std::map<long int, long int>::iterator bufIter = aggrBuffer.begin();
	while(bufIter != aggrBuffer.end())
	{
		if(bufIter->first == pseudoLeaf)
		{
			bufferStack = vis->GALVisitor_CreateBufferStack(reinterpret_cast<BlockStack*>(bufIter->second),goingLeft,aggrSize);
			break;
		}
		bufIter++;
	}
	//create an entry in the buffer for this 'node' is not found
	if(bufIter == aggrBuffer.end())
	{
		bufferStack = vis->GALVisitor_CreateBufferStack(NULL,goingLeft,aggrSize);
		std::pair<std::map<long int, long int>::iterator,bool> ret = aggrBuffer.insert(std::make_pair(pseudoLeaf,reinterpret_cast<long int>(bufferStack)));
		assert(ret.second);
		bufIter=ret.first;
	}

	if(aggrSize >= aggrBlockSize)
	{
		vis->GALVisitor_SetAsCurrentBlockStack2(bufferStack);
		msgTraverse.l.clear();
		msgTraverseRight.l.clear();
		blockId = vis->GALVisitor_GetBufferData(msgTraverse.l, msgTraverseRight.l);
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
void GAL::GAL_TraverseHelper_SendCompressedMessages(GALVisitor* vis)
{
	//search for the entry in the buffer for this 'node'
	std::map<long int, long int>::iterator bufIter = aggrBuffer.begin();
	while(bufIter != aggrBuffer.end())
	{
		MsgTraverse msgTraverse, msgTraverseRight;
		TBlockId blockId;
		GAL_Vertex* pLeaf = reinterpret_cast<GAL_Vertex*>(bufIter->first);
		vis->GALVisitor_SetAsCurrentBlockStack2(reinterpret_cast<BlockStack*>(bufIter->second));
		blockId = vis->GALVisitor_GetBufferData(msgTraverse.l, msgTraverseRight.l);
		if(msgTraverse.l.size() > 0)
		{
			msgTraverse.blkStart = blockId;
			msgTraverse.pRoot = reinterpret_cast<long int>(pLeaf->leftChild);
			msgTraverse.pLeaf = bufIter->first;
			if(pLeaf->rightChild)
			{
				msgTraverse.pSibling = reinterpret_cast<long int>(pLeaf->rightChild);
				msgTraverse.siblingDesc = pLeaf->rightDesc;
			}
			else
			{
				msgTraverse.pSibling = static_cast<long int>(0);
			}
			GAL_SendMessage((pLeaf->leftDesc),MESSAGE_TRAVERSE, &msgTraverse);
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
			GAL_SendMessage((pLeaf->rightDesc),MESSAGE_TRAVERSE, &msgTraverseRight);
		}
#ifdef STATISTICS		

				int stageNum  = PIPELINE_STAGE_NUM(pLeaf->level, numPipelineBuffers, subtreeHeight);
				if(stageNum <= numPipelineBuffers)
					bufferStage[stageNum] += msgTraverse.l.size() + msgTraverseRight.l.size();
				//printf("node->level:%d, stagenum %d numPipelinebuffers:%d bufCount:%d\n",pLeaf->level, stageNum, numPipelineBuffers, msgTraverse.l.size());	
#endif
		aggrBuffer.erase(bufIter);
		bufIter++;
	}

	return;	
}

/* Description: This function is called to handle MESSAGE_TRAVERSE_BACKWARD message containing encoding of multiple blocks. 
 * 		The object containing pseudoLeaf is removed from the list as well.
 * Parameters:  vis = reference to visitor object.
 *             blkIdSet - OUT- set containing IDs of the block whose elements were aggregated and composed as a single block. 
 * Return Value: status of traversal
 */

int GAL::GALHelper_HandleMessageTraverseBackward_Multiple(GAL_Vertex* parentNode, GALVisitor* vis, TCBSet& blkIdSet)
{
	int ret = 1, status = STATUS_SUCCESS;
	bool pipelineFilled = false;
	TCBSet::iterator sIter = blkIdSet.begin();

#ifdef PERFCOUNTERS
	assert(compTimer.IsRunning());
#endif
	
	for(;sIter!=blkIdSet.end();sIter++)
	{
		vis->GALVisitor_SetAsCurrentBlockStack2(*sIter);
		TBlockId curBlockId = (*sIter)->bStackId;
		status = GALHelper_HandleMessageTraverseBackward(parentNode,vis,curBlockId);
		if(status != 1)
		{
#ifdef PERFCOUNTERS
			if(!(compTimer.IsRunning()))
				compTimer.Start();
#endif
			ret = STATUS_TRAVERSE_INCOMPLETE;
			continue;
		}
		else
		{
			int re = vis->GALVisitor_RemoveBlockStack();		
			if(re == -1)
				printf("Debug 11\n");
		}
	}
	
	if((ret ==1) && (vis->GALVisitor_GetNumberOfWorkItems() == 0))
	{
		return ret;
	}
	else
		return STATUS_FAILURE;
	
}

#endif

bool GAL::GALHelper_RemoveFromFragTable(GAL_Vertex* fragger, long int curBlockId, TBlockId& fragBlkId, bool forceErase)
{
#if 0
	FragTableIter fIter = (fragger->fragTable).begin();
	fragBlkId.second = INVALID_BLOCK_ID;
	std::list<FragBlock>::iterator fragBlockIter;

	for(;fIter!=(fragger->fragTable).end();fIter++)
	{
		//if(fIter->fragVertex == fragger)
		{
			fragBlockIter = (fIter->fragBlocks).begin();
			while(fragBlockIter != (fIter->fragBlocks).end())
			{
				std::set<long int>::iterator tmpIter = (fragBlockIter->uniqBlkIds).find(curBlockId);
				if(tmpIter != (fragBlockIter->uniqBlkIds).end())
				{
					(fragBlockIter->uniqBlkIds).erase(tmpIter);
					break;
				}
				fragBlockIter++;
			}
			break;
		}
	}

	assert(fIter != (fragger->fragTable).end());
	fragBlkId = fragBlockIter->inBlkId;
	
	if(((fragBlockIter->uniqBlkIds).size() == 0) || forceErase)
	{
		//if(forceErase)
		{
			//assert((fragBlockIter->uniqBlkIds).size() == 1);
			(fIter->fragBlocks).erase(fragBlockIter);
		}
		if(((fIter->fragBlocks).size()) == 0)
			(fragger->fragTable).erase(fIter);		
		return true;
	}
	else
		return false;
#else
	return false;
#endif
}

bool GAL::GALHelper_FindAndRemoveFromFragTable(GAL_Vertex* node, long int curBlkId, TBlockId& fragBlkId)	
{
#if 0
	std::list<FragBlock>::iterator fragBlockIter;
	bool ret = false;
		
	FragTableIter fIter = (node->fragTable).begin();
	std::set<long int>::iterator tmpIter;
	for(;fIter!=(node->fragTable).end();fIter++)
	{
		//if(fIter->fragVertex == node)
		{
			fragBlockIter = (fIter->fragBlocks).begin();
			while(fragBlockIter != (fIter->fragBlocks).end())
			{
				std::set<long int>::iterator tmpIter = (fragBlockIter->uniqBlkIds).find(curBlkId);
				if(tmpIter != (fragBlockIter->uniqBlkIds).end())
				{
					fragBlkId = fragBlockIter->inBlkId;
					(fragBlockIter->uniqBlkIds).erase(tmpIter);
					if((fragBlockIter->uniqBlkIds).size() == 0)
					{	
						ret = true;
						(fIter->fragBlocks).erase(fragBlockIter);
						if((fIter->fragBlocks).size() == 0)
							(node->fragTable).erase(fIter);
					}
					break;
				}
				fragBlockIter++;
			}
			break;
		}
	}
	return ret;
#else
	/*std::list<FragBlock>::iterator fragBlockIter = (node->fragTable).begin();
	bool ret = false;
	std::set<long int>::iterator tmpIter;
	for(;fragBlockIter!=(node->fragTable).end();fragBlockIter++)
	{
		std::set<long int>::iterator tmpIter = (fragBlockIter->uniqBlkIds).find(curBlkId);
		if(tmpIter != (fragBlockIter->uniqBlkIds).end())
		{
					fragBlkId = fragBlockIter->inBlkId;
					(fragBlockIter->uniqBlkIds).erase(tmpIter);
					if((fragBlockIter->uniqBlkIds).size() == 0)
					{	
						ret = true;
						(node->fragTable).erase(fragBlockIter);
					}
					break;
		}
	}
	return ret;*/
	return false;

#endif
}

int GAL::GALHelper_DeAggregateBlockAndSend(GALVisitor* vis, GAL_Vertex** pRoot, MsgTraverse& msgTraverse, bool loadBalanced)
{
	int ret = STATUS_TRAVERSE_COMPLETE;
	bool traverseIncomplete = false;
	bool siblingTraversalPending = false;
	int startIndx = 0, endIndx = 0;
	
	GAL_Vertex* tmpVertex = *pRoot;
	GAL_PseudoRoot* pSibling = reinterpret_cast<GAL_PseudoRoot*>(msgTraverse.pSibling);
 	int siblingDesc;
	if(pSibling != 0)
		siblingDesc = msgTraverse.siblingDesc;
	BlockStackListIter curBStack = vis->GALVisitor_GetCurrentBlockStack();

#ifdef MESSAGE_AGGREGATION
	int blockSize = GetOpt_BlockSize();

	std::list<BlockStack*> tmpFTable;
	GAL_Vertex* nextNodeToVisit=NULL;

	if(msgTraverse.l.size() > blockSize)
	{
		int totalBlocks = msgTraverse.l.size()/blockSize;
		if(msgTraverse.l.size()%blockSize)
			totalBlocks+=1;
		while(totalBlocks)
		{
			endIndx = startIndx + blockSize;
			if(endIndx > msgTraverse.l.size())
				endIndx = msgTraverse.l.size();
			TIndices workBlock(endIndx-startIndx);
			int j=0;
			for(int i=startIndx;i<endIndx;i++,j++)
				workBlock[j]=(msgTraverse.l.begin()+i)->index;
			
			BlockStack* fragBStack = vis->GALVisitor_CreateBlockStack(workBlock, tmpVertex, NULL, 2, curBStack);
			busyTime.Start();
			ret = GAL_TraverseHelper(vis,tmpVertex, NULL);
			busyTime.Stop();
			
			if(ret == STATUS_TRAVERSE_INCOMPLETE)
			{
#ifdef PERFCOUNTERS
				assert(!compTimer.IsRunning());
				compTimer.Start();
#endif
				traverseIncomplete = true;
			}
			else if(ret == STATUS_TRAVERSE_COMPLETE)
			{
				if(!pSibling || (pSibling && (siblingDesc != procRank)) || (traverseIncomplete) )
				{
					int nextNodeProc=0;
					vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
					vis->GALVisitor_RemoveBlockStack();
				}
				else
				{
					tmpFTable.push_front(fragBStack);
				}
			}
					
			startIndx += blockSize;
			totalBlocks--;
		}

		if(traverseIncomplete)
		{
			std::list<BlockStack*>::iterator fIter = tmpFTable.begin();
			for(;fIter != (tmpFTable).end();fIter++)
			{
				vis->GALVisitor_SetAsCurrentBlockStack2(*fIter);
				vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
				vis->GALVisitor_RemoveBlockStack();
			}
		}

		if(!traverseIncomplete)
		{	
			if(pSibling)
			{
				if(siblingDesc == procRank)	
				{
					//vis->GALVisitor_SetAsCurrentBlockStack2(curBStack);
					int nextNodeProc = 0;
					std::list<BlockStack*>::iterator fIter = tmpFTable.begin();
					for(;fIter != (tmpFTable).end();fIter++)
					{
						vis->GALVisitor_SetAsCurrentBlockStack2(*fIter);
						busyTime.Start();
						ret = GAL_TraverseHelper(vis, pSibling, NULL);
						busyTime.Stop();
						if(ret == STATUS_TRAVERSE_COMPLETE)
						{
							vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
							vis->GALVisitor_RemoveBlockStack();
						}
						else if(ret == STATUS_TRAVERSE_INCOMPLETE)
						{
#ifdef PERFCOUNTERS
							assert(!compTimer.IsRunning());
							compTimer.Start();
#endif
							traverseIncomplete = true;
						}
					}
					tmpVertex = pSibling;
					if(!traverseIncomplete)
					{
						pSibling = NULL;
					}
				}
				else
				{
					siblingTraversalPending = true;
				}
			}
		}

		if(siblingTraversalPending)
		{
#ifdef PERFCOUNTERS
			compTimer.Stop();
#endif
			int nextNodeProc=0;
			assert(tmpVertex == *pRoot);
			MsgTraverse tmp;
			tmp.pLeaf = reinterpret_cast<long int>(((GAL_Vertex*)(*pRoot))->parent);
			tmp.pRoot = reinterpret_cast<long int>(pSibling);
			GAL_PseudoRoot* psRoot = (GAL_PseudoRoot *)(*pRoot);
			tmp.pSibling = 0;//reinterpret_cast<long int>(psRoot->pSibling);
			tmp.blkStart = msgTraverse.blkStart;
			vis->GALVisitor_SetAsCurrentBlockStack2(curBStack);
			vis->GALVisitor_GetLocalData(tmp.l); //msgTraverse.blkStart is different from local blkStart returned by GetLocalData
			vis->GALVisitor_RemoveBlockStack();	
			GAL_SendMessage(siblingDesc,MESSAGE_TRAVERSE, &tmp);
			traverseIncomplete = true;
		}

		if(traverseIncomplete)
		{
#ifdef PERFCOUNTERS
			if(compTimer.IsRunning())
				compTimer.Stop();
#endif
			ret = STATUS_TRAVERSE_INCOMPLETE;
		}
		else
		{
			int nextNodeProc=0;
			vis->GALVisitor_DeleteFromSuperBlock(tmpVertex, &nextNodeToVisit, nextNodeProc);
			assert(ret == STATUS_TRAVERSE_COMPLETE);
			msgTraverse.l.clear();
			vis->GALVisitor_SetAsCurrentBlockStack2(curBStack);
			vis->GALVisitor_GetLocalData(msgTraverse.l);
			*pRoot = tmpVertex;
		}
	}
	else
#endif
	{
		busyTime.Start();
		ret = GAL_TraverseHelper(vis,*pRoot, pSibling);
		busyTime.Stop();
		if(ret == STATUS_TRAVERSE_COMPLETE)
		{
			if(pSibling)
			{
				GAL_Vertex* nextNodeToVisit=NULL;
				int nextNodeProc;
				TBlockId blkStart = vis->GALVisitor_DeleteFromSuperBlock(*pRoot, &nextNodeToVisit, nextNodeProc);
				msgTraverse.l.clear();
				vis->GALVisitor_GetLocalData(msgTraverse.l);
				if((msgTraverse.siblingDesc) != procRank)
				{
#ifdef PERFCOUNTERS
					if(tmpVertex->level == (2*subtreeHeight))
						leafStageTimer.Stop();
					compTimer.Stop();
#endif
					//sending entire compressed data to sibling
					msgTraverse.pLeaf = msgTraverse.pLeaf;
					msgTraverse.pRoot = msgTraverse.pSibling;
					msgTraverse.pSibling = static_cast<long int>(0);
					msgTraverse.siblingDesc = msgTraverse.siblingDesc;//dummy entry
					msgTraverse.blkStart = blkStart;
					if(!loadBalanced)
					{
						int re = vis->GALVisitor_RemoveBlockStack();	
							if(re == -1)
								printf("Debug 1\n");
					}
					GAL_SendMessage((msgTraverse.siblingDesc),MESSAGE_TRAVERSE, &msgTraverse);
					ret = STATUS_TRAVERSE_INCOMPLETE;
				}
				else
				{
					*pRoot = pSibling;
					vis->GALVisitor_AddToSuperBlock(*pRoot, msgTraverse.l, msgTraverse.blkStart, NULL, 0);
					if(!loadBalanced)
					vis->GALVisitor_UpdateCurrentBlockStackId(*pRoot);
					busyTime.Start();
					ret = GAL_TraverseHelper(vis,*pRoot,NULL);
					busyTime.Stop();
				}
			}
	
			if(ret == STATUS_TRAVERSE_COMPLETE)
			{
				GAL_Vertex* nextNodeToVisit=NULL;
				int nextNodeProc=0;
				TBlockId blkStart = vis->GALVisitor_DeleteFromSuperBlock(*pRoot, &nextNodeToVisit, nextNodeProc);	
				msgTraverse.l.clear();
				vis->GALVisitor_GetLocalData(msgTraverse.l);
			}

		}
	}
	
	return ret;
}

int GAL::GetOpt_BlockSize()
{
	if(clopts)
	{
		return clopts->blkSize;
	}
	else
	{
		return BLOCK_SIZE;
	}
}

void GAL::SetOpt_SubtreeHeight(int st)
{
	if(procRank == 0)
		return;

	if(st < 5)
		return;
	if(clopts)
	{
		clopts->subtreeHeight = st;
	}
	else
	{
		assert(0);
	}
}

int GAL::GetOpt_SubtreeHeight()
{
	if(clopts)
	{
		return clopts->subtreeHeight;
	}
	else
	{
		return SUBTREE_HEIGHT;
	}
}

#ifdef MERGE_DEGEN_TREES
int GAL::GetOpt_DegentreeVCount()
{
	if(clopts)
	{
		return clopts->dgentree_vcount;
	}
	else
	{
		return DGENTREE_VCOUNT;
	}
}
#endif

#ifdef SPAD_2
int GAL::GetOpt_NumReplicatedSubtrees()
{
	if(clopts)
	{
		return clopts->numReplicatedSubtrees;
	}
	else
	{
		return 0;
	}
}
#endif

int GAL::GetOpt_BlockIntakeLimit()
{
	if(clopts)
	{
		return clopts->blkIntakeLimit;
	}
	else
	{
		return BLOCK_INTAKE_LIMIT;
	}
}

#ifdef MESSAGE_AGGREGATION
int GAL::GetOpt_NumPipelineBuffers()
{
	if(clopts)
	{
		return clopts->numBuffers;
	}
	else
	{
		return NUM_PIPELINE_BUFFERS;
	}
}

int GAL::GetOpt_PipelineBufferSizes(int stageNum)
{
	if(clopts)
	{
		assert(stageNum < (clopts->pipelineBufferSizes).size());
		return clopts->pipelineBufferSizes[stageNum];
	}
	else
	{
		if(stageNum == 0)
			return PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,1);
		else if (stageNum == 1)
			return  PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,2);
		else if (stageNum == 2)
			return PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,3);
		else
			assert(0);
	}
}
#endif
void GAL::GAL_Synchronize()
{
	//synchronize(pg);
	synchronize((*g).process_group());
}

bool GAL::MedianSplit(GAL_Vertex* intNode, int height, TIndices& pointRefs, TIndices& leftPoints, TIndices& rightPoints, Box& leftBox, Box& rightBox)
{
	// Find the longest axis, this will be the splitting axis.
	float length = (float)0.0;
	int axis = 0;
	for (int i = 0; i < 3; ++i)
	{
		float tmp;
		tmp = intNode->box.max[i] - intNode->box.min[i];
		if (tmp > length)
		{
			length = tmp;
			axis = i;
		}
	}
	float split_plane = (intNode->box.max[axis] + intNode->box.min[axis]) * (float)0.5;

	leftBox = intNode->box;
	rightBox = intNode->box;
	leftBox.max[axis] = split_plane;
	rightBox.min[axis] = split_plane;
	leftBox.calcDimensions();
	rightBox.calcDimensions();

	TriangleVector::iterator tmpIter = points->begin();
	for(int i=0;i<pointRefs.size();i++)
	{
		if(leftBox.testTriangle(&(*(tmpIter + pointRefs[i]))))
		{
			leftPoints.push_back(pointRefs[i]);
		}
		if(rightBox.testTriangle(&(*(tmpIter + pointRefs[i]))))
		{
			rightPoints.push_back(pointRefs[i]);
		}
	}

	if((pointRefs.size () == leftPoints.size ()) && (pointRefs.size () == rightPoints.size ()))
	{
		TriangleVector::iterator piter = points->begin();
		for(int i=0;i<pointRefs.size();i++)
			(intNode->triangles).push_back(*(piter+pointRefs[i]));
		pointRefs.erase(pointRefs.begin(),pointRefs.end());

#ifdef TRAVERSAL_PROFILE
		std::pair<std::map<int,long int>::iterator,bool> ret;
		ret = numLeavesAtHeight.insert(std::pair<int,long int>(height,1) );
		if(ret.second==false)
			numLeavesAtHeight[height] += 1;
#endif
		return false;
	}
	
	return true;
}

bool GAL::SAHSplit(GAL_Vertex* intNode, int height, TIndices& pointRefs, TIndices& leftPoints, TIndices& rightPoints, Box& leftBox, Box& rightBox)
{
	int split_axis = (height) % 3;
	float best_split_plane = (float)0.0;
	float traversal_cost = (float)0.3;
	float intersect_cost = (float)1.0;
	float best_cost = (float)HUGE_VAL;
	std::set<float>  split_planes; // Set of split planes already tried.
	std::pair <std::set<float> ::iterator, bool> return_value;
	float parent_area = (float)1.0 / intNode->getArea();

	// Determine the best split plane to use.
	// Loop over the triangles in this node, the split planes are selected as the
	// current split axis of each primitive.
	TriangleVector::iterator piter = points->begin();
	//for (piter = points.begin();piter!=points.end();piter++)
	for (int m=0;m<pointRefs.size();m++)
	{
		float split_plane = (float)HUGE_VAL;
		// Find the smallest of the vertices of this triangle.
		for (int j = 0; j < 3; ++j)
		{
			if ((piter+pointRefs[m])->m_vertices[j][split_axis] < split_plane)
			{
				// Make sure that this split plane lies inside of the node.  Because triangles can
				// span multiple nodes, this check is necessary.
				if (((piter+pointRefs[m])->m_vertices[j][split_axis] < intNode->box.max[split_axis]) &&
						((piter+pointRefs[m])->m_vertices[j][split_axis] > intNode->box.min[split_axis]))
				{
					split_plane = (piter+pointRefs[m])->m_vertices[j][split_axis];
				}
			}
		}

		if (split_plane != HUGE_VAL)
		{
			return_value = split_planes.insert (split_plane);
			if (return_value.second)
			{
				// We have not yet evaluated this split plane.
				// Make two new nodes that would be children of this node using the current split plane.
				int left_count=0,right_count=0;
				leftBox = intNode->box;
				rightBox = intNode->box;
				leftBox.max[split_axis] = split_plane;
				rightBox.min[split_axis] = split_plane;

				leftBox.calcDimensions();
				rightBox.calcDimensions();

				TriangleVector::iterator tmpIter = points->begin();
				for(int i=0;i<pointRefs.size();i++)
				{
					if(leftBox.testTriangle(&(*(tmpIter + pointRefs[i]))))
					{
						left_count++;
					}
					if(rightBox.testTriangle(&(*(tmpIter + pointRefs[i]))))
					{
						right_count++;
					}
				}

				float left_prob = leftBox.getArea() * parent_area;
				float right_prob = rightBox.getArea() * parent_area;

				float cost = traversal_cost + intersect_cost * (left_prob * left_count + right_prob * right_count);
				if (cost < best_cost)
				{
					best_cost = cost;
					best_split_plane = split_plane;
				}
			}
		}
	}
	

	// Now that the best split plane has been found, determine if this cost is less than simply raytracing the node itself.
	if (best_cost < (intersect_cost * pointRefs.size ()))
	{
		leftBox = intNode->box;
		leftBox.max[split_axis] = best_split_plane;
		leftBox.calcDimensions();
		rightBox = intNode->box;
		rightBox.min[split_axis] = best_split_plane;
		rightBox.calcDimensions();

		TriangleVector::iterator tmpIter = points->begin();
		for(int i=0;i<pointRefs.size();i++)
		{
			if(leftBox.testTriangle(&(*(tmpIter + pointRefs[i]))))
			{
				leftPoints.push_back(pointRefs[i]);
			}
			if(rightBox.testTriangle(&(*(tmpIter + pointRefs[i]))))
			{
				rightPoints.push_back(pointRefs[i]);
			}
		}
	}
	else
	{
		//stop recursing.
		TriangleVector::iterator piter = points->begin();
		for(int i=0;i<pointRefs.size();i++)
			(intNode->triangles).push_back(*(piter+pointRefs[i]));
		pointRefs.erase(pointRefs.begin(),pointRefs.end());
#ifdef TRAVERSAL_PROFILE
		std::pair<std::map<int,long int>::iterator,bool> ret;
		ret = numLeavesAtHeight.insert(std::pair<int,long int>(height,1) );
		if(ret.second==false)
			numLeavesAtHeight[height] += 1;
#endif

		return false;
	}

	if((pointRefs.size () == leftPoints.size ()) && (pointRefs.size () == rightPoints.size ()))
	{
		TriangleVector::iterator piter = points->begin();
		for(int i=0;i<pointRefs.size();i++)
			(intNode->triangles).push_back(*(piter+pointRefs[i]));
		pointRefs.erase(pointRefs.begin(),pointRefs.end());

#ifdef TRAVERSAL_PROFILE
		std::pair<std::map<int,long int>::iterator,bool> ret;
		ret = numLeavesAtHeight.insert(std::pair<int,long int>(height,1) );
		if(ret.second==false)
			numLeavesAtHeight[height] += 1;
#endif
		return false;
	}

	return true;
}

template<typename T>
void GAL_BroadCastObject(T& obj,GAL* g)
{
	for(int i=0;i<g->numProcs;i++)
	{
		if(i != g->procRank)
		{
			send_oob(g->pg,i,MESSAGE_DATA,obj); 
		}
	}
	return;
}

bool GAL_ReceiveObject(Photon& obj,GAL* g)
{
	PBGL_oobMessage msg;
	bool dataReceived = false;
	bool done = false;
	while(!done)
	{
		PBGL_AsyncMsg pollMsg = (g->pg).poll();
		if(!pollMsg)
			continue;
		else
			msg = pollMsg.get();
		switch(msg.second)
		{
				case MESSAGE_DATA:
				{
					
				     	receive_oob(g->pg, msg.first, msg.second, obj);
					dataReceived = true;
					done = true;
				}
				break;
				case MESSAGE_DONEKDTREE:
				{
					int i=0;
				     	receive_oob(g->pg, msg.first, msg.second, i);
					dataReceived = false;
					done =true;
				}
				break;
				default:done=true;break;	 
		}
	}
	return dataReceived;
}

void GAL_BroadCastMessage(GAL* g, int msg)
{
	for(int i=0;i<g->numProcs;i++)
	{
		if(i != g->procRank)
		{
			send_oob(g->pg,i,msg,1); 
		}
	}
	return;
}

template<typename T>
void GAL::GAL_Test(T& obj)
{
}

#ifdef PERFCOUNTERS
void GAL::GAL_StartComputeTimer()
{
	compTimer.Start();
}
void GAL::GAL_StopComputeTimer()
{
	compTimer.Stop();
}
#endif

void GAL::GAL_UpdateLocalNodeCount(int procId, int nodeCount)
{
	assert(procId != procRank);
	nodeCountTable[procId] += nodeCount;
}

int GAL::GAL_GetNextProcessId(int subtreeHeight, long int pLeaf)
{

	if(numProcs == 1)
		return procRank;
	int nextprocessid =0;
	/*std::pair<std::map<long int, int>::iterator, bool> ret = pLeafMap.insert(std::make_pair(pLeaf,0));
	if(ret.second == false)
	{
		return (ret.first)->second;
	}
	else*/
	{
		do
		{
			nextprocessid = ((procRank)* 2 + 1 + procCount) % numProcs;
			procCount++;
			if(procCount > (1<<(subtreeHeight)))
				procCount = 1;
		}while(nextprocessid == procRank);
	}
	//(ret.first)->second = nextprocessid;

	return nextprocessid;
}

void GAL::GALHelper_CountSubtreeNodes(GAL_Vertex* ver, long int& count, bool profilingData, bool isRootSubtree)
{
	int ret = STATUS_TRAVERSE_COMPLETE;
	if(profilingData)
	{
#ifdef TRAVERSAL_PROFILE
		count += ver->pointsVisited;
#else
		count++;
#endif
	}
	else
		count++;
	assert(ver != NULL);

	assert((ver->desc).owner ==procRank); 
	if((ver->leftChild == NULL) && (ver->rightChild==NULL))
	{
		return;
	}
	
	if(ver->leftChild)
	{
		if(ver->leftDesc == procRank)
		{
			/*if((ver->leftChild->pseudoRoot) && ((ver->leftChild->leftChild && (ver->leftChild->leftDesc != procRank)) || (ver->leftChild->rightChild && (ver->leftChild->rightDesc != procRank))))
				return;*/
			if(isRootSubtree && GALHelper_IsStage(reinterpret_cast<long int>(ver->leftChild)))
				return;
			GALHelper_CountSubtreeNodes(ver->leftChild,count,profilingData, isRootSubtree);
		}
	}
	if(ver->rightChild)
	{
		if(ver->rightDesc == procRank)
		{
			/*if((ver->rightChild->pseudoRoot) && ((ver->rightChild->leftChild && (ver->rightChild->leftDesc != procRank)) || (ver->rightChild->rightChild && (ver->rightChild->rightDesc != procRank))))
				return;*/
			if(isRootSubtree && GALHelper_IsStage(reinterpret_cast<long int>(ver->rightChild)))
				return;
			GALHelper_CountSubtreeNodes(ver->rightChild,count,profilingData,isRootSubtree);	
		}
	}
			
		
}


#ifdef LOAD_BALANCE
GAL_Vertex* GAL::GAL_GetLocalVertex(long int localNode)
{
	std::vector<GAL_Vertex*>::iterator iter = pseudoLeaves.begin();
	GAL_Vertex* ret = NULL;
	for(iter;iter != pseudoLeaves.end();iter++)
	{
		if(((*iter)->desc).local == localNode)
		{
			ret = (*iter);
			break;
		}
	}
	assert(ret);
	return ret;
}

int GAL::GAL_Aux_UpdatePLeaves(MsgUpdatePLeaves& msg, int pid)
{
	int ret = STATUS_SUCCESS;
	std::vector<GAL_Vertex*>::iterator leafTableIter = pseudoLeaves.begin();
	std::vector<MsgUpdatePLeaf>::iterator msgIter = msg.vPLeaves.begin();

	if(numUpdatesRequired > 0)
		ret = STATUS_FAILURE;
	int i=0;
	for(msgIter = msg.vPLeaves.begin();msgIter != msg.vPLeaves.end();msgIter++)
	{
		for(leafTableIter=pseudoLeaves.begin();leafTableIter != pseudoLeaves.end();leafTableIter++)
		{
			if(((*leafTableIter)->desc).local == (msgIter->leafDesc).local)
			{
				bool leftUpdateRequired=false, rightUpdateRequired=false;
				//if(msgIter->pLeftChild)
				if(((*leafTableIter)->leftChild == NULL) && msgIter->pLeftChild)
				{
					(*leafTableIter)->leftDesc = msgIter->leftDesc;	
					(*leafTableIter)->leftChild = reinterpret_cast<GAL_Vertex*>(msgIter->pLeftChild);
					leftUpdateRequired = true;	
				}
				//if(msgIter->pRightChild)
				if(((*leafTableIter)->rightChild == NULL) && (msgIter->pRightChild))
				{
					(*leafTableIter)->rightDesc = msgIter->rightDesc;	
					(*leafTableIter)->rightChild = reinterpret_cast<GAL_Vertex*>(msgIter->pRightChild);	
					rightUpdateRequired = true;	
				}

				MsgUpdatePRoot msgUpdatePRoot;
				msgUpdatePRoot.pLeaf = reinterpret_cast<long int>((*leafTableIter));
				msgUpdatePRoot.leafDesc = procRank;
				if(leftUpdateRequired && msgIter->pLeftChild)
				{
					msgUpdatePRoot.pRoot = msgIter->pLeftChild;
					if((msgIter->leftDesc) != procRank)
					{
						//printf("process:%d, leftowner:%d rightowner:%d \n",pid,(msgIter->leftDesc).owner, (msgIter->rightDesc).owner);
						//printf("process %d sending UPDATEROOT to process %d\n",process_id(g->process_group()),(msgIter->leftDesc).owner);
						GAL_SendMessage((msgIter->leftDesc),MESSAGE_UPDATE_PROOT,&msgUpdatePRoot); 
					}
					else
					{
						GAL_PseudoRoot* child = reinterpret_cast<GAL_PseudoRoot *>(msgIter->pLeftChild);
						(child->parents).push_back(std::make_pair(procRank, reinterpret_cast<long int>((*leafTableIter))));
					}
				}

				if(rightUpdateRequired && msgIter->pRightChild)
				{
					msgUpdatePRoot.pRoot = msgIter->pRightChild;

					if((msgIter->rightDesc) != procRank)
					{
						//printf("process %d sending UPDATEROOT to process %d\n",process_id(g->process_group()),(msgIter->rightDesc).owner);
						GAL_SendMessage((msgIter->rightDesc),MESSAGE_UPDATE_PROOT,&msgUpdatePRoot); 
					}
					else
					{
						GAL_PseudoRoot* child = reinterpret_cast<GAL_PseudoRoot *>(msgIter->pRightChild);
						(child->parents).push_back(std::make_pair(procRank, reinterpret_cast<long int>((*leafTableIter))));
					}
				}
				break;
			}
		}
	}
	
	return ret;
}
using boost::parallel::all_gather;
void GAL::AllGather(std::vector<Photon>& localPhotons, std::vector<Photon>& globalPhotons)
{
	//important to use the graph's process group here. pg is a reference to process group object that is local to the process
	all_gather((*g).process_group(), localPhotons.begin(), localPhotons.end(), globalPhotons);
}

#endif
void GAL::GAL_AggregateResults(uint64_t localSum, double localTime, uint64_t& totalSum, double& totalTime)
{
	reduce(communicator(pg),localSum, totalSum, std::plus<long int>(),0);
	reduce(communicator(pg),localTime, totalTime, boost::parallel::maximum<int>(),0);
	if(procRank == 0)
	{
		//printf("nodes_traversed:%lld time_consumed:%f\n",totalSum,totalTime/CLOCKS_PER_SEC);
		printf("nodes traversed:%lld\n",totalSum);
		printf("time consumed:%f\n",totalTime/CLOCKS_PER_SEC);
	}
	double maxTraversalTime;
	reduce(communicator(pg),traversalTime, maxTraversalTime, boost::parallel::maximum<double>(),0);
	//printf("Traversal time %f busyTime %f\n",traversalTime, busyTime.GetTotTime()/(double)CLOCKS_PER_SEC);
	if(procRank == 0)
		printf("Traversal time: %f seconds\n",maxTraversalTime);

}

void GAL::GAL_AggregateResults_Generic(long int localSum, long int& totalSum)
{
	reduce(communicator(pg),localSum, totalSum, std::plus<long int>(),0);
}
#ifdef MERGE_DEGEN_TREES
void GAL::GALHelper_DeleteSubtree(GAL_Vertex* ver)
{
		assert(ver != NULL);

		GAL_Vertex* node = ver;
		assert((node->desc).owner ==procRank); 
		
		if((node->leftChild == NULL) && (node->rightChild == NULL))
		{
			GAL_DeleteVertex(node);
			return;
		}
		
		if(node->leftChild)
		{
			if(node->leftDesc == procRank)
			{
					GALHelper_DeleteSubtree(node->leftChild);
			}
		}
		if(node->rightChild)
		{
			if(node->rightDesc == procRank)
			{
					GALHelper_DeleteSubtree(node->rightChild);
			}
		}
		GAL_DeleteVertex(node);
}
#endif

void GAL::print_treetofile(std::ofstream& fp)
{
	print_preorder(rootNode, fp);
}

void GAL::print_preorder(GAL_Vertex* node, std::ofstream& fp)
{
	static int label;
	fp<<label++<<":"<<node->box;
	//fp<<label++<<":"<<(int)node->level<<std::endl;
	//std::cout<<label++<<":"<<(int)node->level<<std::endl;
	//fp<<"----"<<std::endl;
	/*if(node->level == 9)
		printf("debug break\n");*/
	if(node->leftChild)
		print_preorder(node->leftChild,fp);

	if(node->rightChild)
		print_preorder(node->rightChild,fp);
}

void GAL::GALHelper_GetBlockOccupancyFactor(GAL_Vertex* ver, double& count)
{
		int ret = STATUS_TRAVERSE_COMPLETE;
#ifdef TRAVERSAL_PROFILE
		count += ver->blockOccupancyFactor;
#endif
		assert(ver != NULL);

		assert((ver->desc).owner ==procRank); 
		
		if((ver->leftChild == NULL) && (ver->rightChild==NULL))
		{
			return;
		}

		if(ver->leftChild)
		{
			if(ver->leftDesc == procRank)
			{
				GAL_Vertex* tmp=ver->leftChild;
				//if(!(tmp->pseudoRoot) || (tmp->pseudoRoot && (tmp->leftChild==NULL) && (tmp->rightChild==NULL)))
				{
					GALHelper_GetBlockOccupancyFactor(tmp,count);
				}
			}
		}
		if(ver->rightChild)
		{
			if(ver->rightDesc == procRank)
			{
				GAL_Vertex* tmp=ver->rightChild;
				//if(!(tmp->pseudoRoot) || (tmp->pseudoRoot && (tmp->leftChild==NULL) && (tmp->rightChild==NULL)))
				{
					GALHelper_GetBlockOccupancyFactor(tmp,count);
				}
			}
		}
			
		
}

#ifdef SPAD_2
void GAL::GALHelper_ReplicateSubtrees()
{
	PBGL_oobMessage msg;
	int numLocalReplicas=0;
	bool done=false;
	if(adjacentSubtrees.size() > 0)
		printf("Number of subtrees adjacent to process %d:%d\n", procRank,adjacentSubtrees.size());
	if(procRank==0)
	{
		int status=GALHelper_GetRandomSubtreeAndBroadcast();
		//int status=GALHelper_GetBottleneckSubtreesAndBroadcast(107359,"bneck107359.txt");
		if(status==STATUS_SUCCESS)
		{
			for(int i=1;i<numProcs;i++)
			{
				GAL_SendMessage(i,MESSAGE_READYTOEXIT,&procRank); 
			}
			done = true;
		}
	}
	while (!done)
	{
		PBGL_AsyncMsg pollMsg = pg.poll();
		if(!pollMsg)
		{
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
					MsgReplicateSubtree msgRepSubtree;
					receive_oob(pg, msg.first, msg.second, msgRepReq);
					//printf("%d receiving subtree replication request from %d\n",procRank,msg.first);
					numReplicatedSubtrees = msgRepReq.numSubtreesReplicated;
					GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(msgRepReq.pRoot);
					int childNum = msgRepReq.childNum;
					msgRepSubtree.childNum = msgRepReq.childNum;
					msgRepSubtree.pLeaf = msgRepReq.pLeaf;
					msgRepSubtree.pLeafLabel = msgRepReq.pLeafLabel;
					GALHelper_SaveSubtreeAsString(pRoot, msgRepSubtree.data, msgRepReq.pLeaf, childNum);
					numLocalReplicas++;
					for(int i=0;i<numProcs;i++)
					{
						if(i!=procRank)
						{
							GAL_SendMessage(i,MESSAGE_REPLICATE_SUBTREE, &msgRepSubtree);
						}
					}
					if(numLocalReplicas == numReplicatedSubtrees)
					{
						//printf("process %d exiting \n", procRank);
						done = true;
					}
				}
				break;
				case MESSAGE_REPLICATE_SUBTREE:
				{
					MsgReplicateSubtree msgReplicateSubtree;
					receive_oob(pg, msg.first, msg.second, msgReplicateSubtree);
					//printf("%d received subtree data from %d size %d\n",procRank,msg.first, msgReplicateSubtree.data.size());
					numLocalReplicas++;
					numReplicatedVertices += msgReplicateSubtree.data.size();
					GAL_Vertex* pRoot=NULL;
					GAL_Vertex* pLeaf = NULL;
					std::vector<GAL_Vertex*>::iterator vIter = pseudoLeaves.begin();
					for(;vIter!=pseudoLeaves.end();vIter++)
					{
						if((*vIter)->id == msgReplicateSubtree.pLeafLabel)
						{
							pLeaf = *vIter;
							break;
						}
					}	
					assert(pLeaf);
					//printf("%d replicated subtree at %d(from %d)\n",procRank,pLeaf->label,msg.first);
					GALHelper_ReplicateSubtreeFromString(msgReplicateSubtree,&pRoot, pLeaf,msg.first);
					assert(pRoot!=NULL);
					if(numLocalReplicas == numReplicatedSubtrees)
					{
						done = true;
						//printf("process %d exiting \n", procRank);
					}
				}
				break;
				case MESSAGE_READYTOEXIT:
				{
					int doneProcId;
					receive_oob(pg, msg.first, msg.second, doneProcId);
					done = true;
				}
				break;

				default: break;
			}
		}
	}
	long int numVerticesInRootSubtree=0;
	GALHelper_CountSubtreeNodes(rootNode, numVerticesInRootSubtree, false, true);
	numReplicatedVertices +=numVerticesInRootSubtree;
	if(procRank == 0)
		printf("%d numReplicatedVertices: %ld numVerticesInRootSubtree:%d\n",procRank,numReplicatedVertices, numVerticesInRootSubtree);
	synchronize(pg);
	readyToExit=false;
	readyToExitList.clear();
}

int GAL::GALHelper_GetRandomSubtreeAndBroadcast()
{
	int ret = STATUS_SUCCESS;
	int i=0, size=adjacentSubtrees.size();
	std::vector<SubtreeHeader> candidateSubtrees;
	std::set<int> candidateIndices;
	for(int i=0;i<numReplicatedSubtrees;i++)
	{
		long pLeafIndex = GALHelper_GetRandom(size);
		if(pLeafIndex == size)
			pLeafIndex = size-1;
		candidateIndices.insert(pLeafIndex);
		//candidateIndices.insert(i);
		//printf("pRootIndex:%d\n",pLeafIndex);
	}
	std::set<int>::iterator sIter = candidateIndices.begin();
	for(;sIter!=candidateIndices.end();sIter++)
	{
		std::vector<SubtreeHeader>::iterator pRootIter = adjacentSubtrees.begin();
		for(i=0;pRootIter!=adjacentSubtrees.end();pRootIter++,i++)
		{
			if(i == *sIter)
			{
				candidateSubtrees.push_back(*pRootIter);	
				break;
			}
		}
	}
	numReplicatedSubtrees = candidateSubtrees.size(); 
	printf("Number of replicated subtrees :%d\n",numReplicatedSubtrees);

	std::vector<SubtreeHeader>::iterator subtreeIter = candidateSubtrees.begin();
	for(;subtreeIter!=candidateSubtrees.end();subtreeIter++)
	{
		GAL_Vertex* pLeaf=reinterpret_cast<GAL_Vertex*>(subtreeIter->pLeaf);
		assert(pLeaf != NULL);
		MsgReplicateReq msg;
		msg.childNum = subtreeIter->childNum;
		msg.pRoot = subtreeIter->pRoot;
		msg.pLeaf = subtreeIter->pLeaf;
		msg.pLeafLabel = pLeaf->id;
		msg.numSubtreesReplicated = numReplicatedSubtrees;
		GAL_SendMessage(subtreeIter->pRootDesc,MESSAGE_REPLICATE_REQ,&msg);
		ret = STATUS_FAILURE;
	}
	return ret;
}

long GAL::GALHelper_GetRandom(int size)
{
	unsigned long 
	// max <= RAND_MAX < ULONG_MAX, so this is okay.
	num_bins = (unsigned long) size + 1,
	num_rand = (unsigned long) RAND_MAX + 1,
	bin_size = num_rand / num_bins,
	defect   = num_rand % num_bins;
	long x;
	do {
	x = rand();
	}
	// This is carefully written not to overflow
	while (num_rand - defect <= (unsigned long)x);
	// Truncated division is intentional
	return x/bin_size;
}

void GAL::GALHelper_SaveSubtreeAsString(GAL_Vertex* subtreeRoot, std::vector<std::string>& subtreeStr, long int parentId, int childNum)
{
#if 1
	assert(subtreeRoot != NULL);

	GAL_Vertex* node = subtreeRoot;
	assert((node->desc).owner ==procRank); 

	std::stringstream stroutput;
	int pLeaf = 0;
	if(node->pseudoRoot)
		pLeaf=2; //special value to capture pseudoroot;
	int isLeaf=((node->leftChild==NULL) && (node->rightChild==NULL))?1:0;
	if(node->leftChild && (node->leftDesc!=procRank))
		pLeaf=1;
	if(node->rightChild && (node->rightDesc!=procRank))
		pLeaf=1;
	assert(pLeaf != 1);
	
	stroutput<<node->id<<" "<<parentId<<" "<<childNum<<" "<<(short int)(node->level)<<" "<<pLeaf<<" "<<isLeaf;
	if(pLeaf==1)
	{
		stroutput<<" "<<reinterpret_cast<long int>(node->leftChild)<<" "<<(node->leftDesc)<<" "<<reinterpret_cast<long int>(node->rightChild)<<" "<<(node->rightDesc);
	}
	if(isLeaf)
	{
	}
	else
	{
	}
	subtreeStr.push_back(stroutput.str());

	if(isLeaf)
	{
		return;
	}
	
	if(node->leftChild && (node->leftDesc == procRank))
	{
		GALHelper_SaveSubtreeAsString(node->leftChild,subtreeStr,node->id,0);
	}
	if(node->rightChild && (node->rightDesc == procRank))
	{
		GALHelper_SaveSubtreeAsString(node->rightChild,subtreeStr,node->id,1);
	}
	
	return;

#endif
}	

//IMP: assumption that this function is called only on internal nodes(i.e pseudoleaves) and not on rootNode.
void GAL::GALHelper_ReplicateSubtreeFromString(MsgReplicateSubtree& msgCloneSubtree, GAL_Vertex** pRoot, GAL_Vertex* pLeaf, int pLeafOwner)
{
#if 1
	std::map<long int,long int> repVertexTable;
	int recordNum = 0;
	std::vector<std::string>::iterator iter = msgCloneSubtree.data.begin();
	std::vector<GAL_Vertex*> pLeavesWithMergedTrees;
	
	for(;iter != msgCloneSubtree.data.end();iter++)
	{
		std::stringstream strinput(*iter);
		int childNum, isLeaf=0, isPLeaf=0;
		long int label, parentId;
		short int level;

		recordNum++;
			
		GAL_Vertex *parentNode, *node=NULL;
		strinput >> label;
		strinput >> parentId;
		strinput >> childNum;
		strinput >> level;
		strinput >> isPLeaf;
		strinput >> isLeaf;


		if(recordNum == 1)
		{
			parentNode=pLeaf;
			/*if(childNum==0)
				printf("%d: received request from  %d to create a replicated left subtree at %p\n",procRank,pLeafOwner,pLeaf);
			else
				printf("%d: received request from  %d to create a replicated right subtree at %p\n",procRank,pLeafOwner,pLeaf);*/
			node = GAL_CreateVertex(VTYPE_PSEUDOROOT);
			node->pseudoRoot = true;
			*pRoot=node;
		}
		else
		{
			if((isPLeaf==2)||(isLeaf==1))
			{
				node = GAL_CreateVertex(VTYPE_PSEUDOROOT);
				node->pseudoRoot = true;
			}
			else
				node = GAL_CreateVertex(VTYPE_NORMAL);

			parentNode = reinterpret_cast<GAL_Vertex*>(repVertexTable[parentId]);
		}
		

		node->id=label;
		node->level=level;
		node->parentDesc = procRank;
		node->parent = parentNode;
		if(childNum==0)
		{
			parentNode->leftChild=node;
			parentNode->leftDesc = procRank;
			node->isLeftChild=true;
		}
		else
		{
			parentNode->rightChild=node;
			parentNode->rightDesc = procRank;
			node->isLeftChild=false;
		}

		long int vertexPtr = reinterpret_cast<long int>(node);
		repVertexTable.insert(std::make_pair(label,vertexPtr));	

		if(isPLeaf==1)
			assert(0);
		
		if(isLeaf)
		{
		}
		else
		{
		}
		
		/*if(isPLeaf==1)
		{
			std::map<int,std::vector<long int> > pRootBucket;
			for(int i=0;i<MAX_CHILDREN;i++)
			{
				int childDesc;
				long int pChild;
				strinput >> pChild;
				strinput >> childDesc;
				if(!pChild)
					continue;

				node->childDesc[i] = childDesc;
				node->pChild[i]=reinterpret_cast<BH_Vertex*>(pChild);
				if((childDesc != procRank) && (childDesc != pLeafOwner))
				{
						pRootBucket[childDesc].push_back(pChild);
				}
				else if(childDesc == procRank)
				{
					BH_Vertex* tmp = reinterpret_cast<BH_Vertex*>(pChild);
					assert((tmp->vData)->pseudoRoot);
					tmp->pParent = node;
					tmp->parentDesc = procRank;
				}
				else
				{
					pLeavesWithMergedTrees.push_back(node);
				}
			}
			std::map<int, std::vector<long int> >::iterator iter = pRootBucket.begin(); 
			for(;iter !=pRootBucket.end();iter++)
			{
				//assert(!isRelocated);
				int pRootProcessId = iter->first;
				MsgUpdatePRoot msgUpdatePRoot;
				msgUpdatePRoot.pLeaf = reinterpret_cast<long int>(node);
				msgUpdatePRoot.pRoots = iter->second;
				msgUpdatePRoot.isPLeafRelocated = true;
				GAL_SendMessage(pRootProcessId,MESSAGE_UPDATE_PROOT,&msgUpdatePRoot); 
			}
		}*/	
	}

	/*std::vector<GAL_Vertex*>::iterator viter = pLeavesWithMergedTrees.begin(); 
	for(;viter !=pLeavesWithMergedTrees.end();viter++)
	{
		std::map<int,bool> pRootProcessList;
		MsgUpdateSiblingInfo msgSiblingInfo;

		BH_Vertex* pLeaf = (BH_Vertex*)*viter;
		for(int i=0;i<MAX_CHILDREN;i++)
		{
			if(pLeaf->pChild[i])
			{
				msgSiblingInfo.pSibling[i] = reinterpret_cast<long int>(pLeaf->pChild[i]);
				if(pLeaf->childDesc[i] == procRank)
				{
					msgSiblingInfo.siblingDesc[i] = procRank;
					msgSiblingInfo.pRoot[i] = 0;
				}
				else
				{
					msgSiblingInfo.siblingDesc[i] = pLeaf->childDesc[i];
					msgSiblingInfo.pRoot[i] = reinterpret_cast<long int>(pLeaf->pChild[i]);
					pRootProcessList[pLeaf->childDesc[i]]=true;
				}
			}
			else
			{
				msgSiblingInfo.pRoot[i] = 0;
				msgSiblingInfo.pSibling[i] = 0;
			}
		}
		std::map<int,bool>::iterator iter2 = pRootProcessList.begin();
		for(;iter2 != pRootProcessList.end();iter2++)
		{
			GAL_SendMessage(iter2->first,MESSAGE_UPDATE_SIBLINGINFO,&msgSiblingInfo); 
		}
	}
	pLeavesWithMergedTrees.erase(pLeavesWithMergedTrees.begin(),pLeavesWithMergedTrees.end());*/

	repVertexTable.erase(repVertexTable.begin(),repVertexTable.end());
#endif
}

void GAL::GALHelper_ReadBottleneckDetails(std::ifstream& input, int numBottlenecks, std::vector<SubtreeHeader>& bottlenecks)
{
        while(true) 
	{
		int procID, childNum;
		long int pLeafLabel, pRootLabel;
		input >> procID >> pLeafLabel >> pRootLabel >> childNum;
		SubtreeHeader hdr(pRootLabel, childNum, procID);
		hdr.pLeaf = pLeafLabel;
		bottlenecks.push_back(hdr);
		if(bottlenecks.size() == numBottlenecks)
			break;
        }
}

int GAL::GALHelper_GetBottleneckSubtreesAndBroadcast(int numBottlenecks, char* bneckfile)
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
		GALHelper_ReadBottleneckDetails(input, numBottlenecks, bottlenecks);
	}
	numReplicatedSubtrees = bottlenecks.size(); 
	printf("Number of replicated subtrees :%d\n",numReplicatedSubtrees);

	/*std::vector<SubtreeHeader>::iterator subtreeIter = bottlenecks.begin();
	for(;subtreeIter!=bottlenecks.end();subtreeIter++)
	{
		MsgReplicateReq msg;
		msg.pLeafLabel=subtreeIter->pLeaf;
		msg.childNum = subtreeIter->childNum;
		msg.numSubtreesReplicated = bottlenecks.size();
		msg.pRoot=0;
		msg.pLeaf=0;
		GAL_SendMessage(subtreeIter->pRootDesc,MESSAGE_REPLICATE_REQ,&msg);
		ret = STATUS_FAILURE;
	}*/
	input.close();
	return ret;
}

bool GAL::GALHelper_IsBottleneck(std::vector<SubtreeHeader>& bneck, GAL_Vertex* pRoot)
{
	std::vector<SubtreeHeader>::iterator iter = bneck.begin();
	for(;iter!=bneck.end();iter++)
	{
		int childNum = pRoot->isLeftChild?0:1;
		if((iter->pRoot == pRoot->id) && (iter->pLeaf == ((GAL_PseudoRoot*)pRoot)->pLeafLabel) && (iter->childNum == childNum))
			return true;
	}
	return false;	
}
#endif


bool GAL::GALHelper_IsStage(long int pRoot)
{
#ifdef TRAVERSAL_PROFILE
	std::vector<long int>::iterator iter = pipelineStage.begin();
	for(;iter!=pipelineStage.end();iter++)
	{
		if(*iter==pRoot)
		{
			return true;
		}
	}
#endif
	return false;
}

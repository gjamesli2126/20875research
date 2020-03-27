#include "GAL.h"
#include "CorrelationVisitor.h"
#include "timer.h"
#include<boost/graph/distributed/depth_first_search.hpp>
#include<boost/bind.hpp>

long int labelCount; //counter to assign label values to a node.
std::map<int,int> numNodesAtHeight;


std::vector<GAL_Vertex*> pseudoLeaves;
std::vector<MsgUpdatePLeaf> vPLeaves;

#ifdef PERFCOUNTERS
double timeStep= ((CLOCKS_PER_SEC/(float)(1.0)));
std::vector<int> tpFinishedBlocks;
timer compTimer;
timer workloadTimer;
uint64_t workloadTime;
#endif

#ifdef TRAVERSAL_PROFILE
std::vector<long int> pipelineStage; //pipelineStageId, BlockStats.
#endif
typedef optional<PBGL_oobMessage> PBGL_AsyncMsg;

float mydistance(const Point& a, const Point& b) {
	float	d=0.;
	for(int i = 0; i < DIMENSION; i++) {
		float diff = a.pt[i] - b.pt[i];
		d += diff * diff;
	}
	return sqrt(d);
}

struct DistanceComparator
{
	const Point& item;
	DistanceComparator( const Point& _item ) : item(_item) {}
	bool operator()(const Point& a, const Point& b) {
		return mydistance( item, a ) < mydistance(item, b );
	}
};

bool SortIntPairsBySecond(const std::pair<int, int>& a, const std::pair<int, int>& b)
{
	return a.second < b.second;
}


GAL* GAL::instance = NULL;
GAL* GAL::GAL_GetInstance(mpi_process_group& prg)
{
		if(instance == NULL)
			instance  = new GAL(prg);

		return instance;
}

GAL_Graph& GAL::GAL_GetGraph()
{
#ifdef PARALLEL_BGL
	return *g;
#endif	
}

GAL_Vertex* GAL::GAL_GetRootNode()
{
	return rootNode;
}

GAL_Vertex* GAL::GAL_CreateVertex(TVertexType vType)
{
	numVertices++;
#ifdef PARALLEL_BGL
	labelCount++;
	GAL_Vertex* node; 
	BGLVertexdesc d;
	if(vType == VTYPE_PSEUDOROOT)
	{
		node = new GAL_PseudoRoot();
#ifdef TRAVERSAL_PROFILE
		pipelineStage.push_back(reinterpret_cast<long int>(node));
#endif
	}
	else
	{
		node =  new GAL_Vertex();
	}
	d = add_vertex(*g);
	node->desc = d;
	node->label = labelCount;
	return node;
#endif	
}

void GAL::GAL_SendMessage(int processId, int messageId, void* msg)
{
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
		case MESSAGE_DONEVPTREE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgTerminate *>(msg))); 
					break;
		case MESSAGE_TRAVERSE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgTraverse*>(msg))); 
					break;
		case MESSAGE_TRAVERSE_BACKWARD:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgTraverse*>(msg))); 
					break;
		case MESSAGE_SENDCOMPRESSED:
					send_oob(pg,processId,messageId,*(reinterpret_cast<int*>(msg))); 
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

		default:
			break;
	}
#endif
}

int GAL::GAL_ConstructVPTree(TPointVector& points, int stHeight)
{
#ifdef PARALLEL_BGL
	
	bool done = false;
	int depth = 0;
	subtreeHeight = stHeight;

	//printf("%d: sorted based on depth 0 , %f %f %f %f %f %f %f\n",procRank,points.at(0).pt[0],points.at(1).pt[0],points.at(2).pt[0],points.at(3).pt[0],points.at(4).pt[0],points.at(5).pt[0],points.at(6).pt[0]);
	depth = 0;
	
	MsgBuildSubTree msgBuildSubTree;
	MsgUpdateMinMax msgDoneTree;
	
	//process 0 distributes the left and right subtrees among processes 0 and 1(if present)
	//if(procRank == 0)
	{

		int donebuildsubtree = GAL_BuildSubTree(points,NULL, 0, points.size(), 0, 0, false);
		if(donebuildsubtree == BUILDSUBTREE_SAMEPROCESS)
		{
			MsgTerminate msgTerminate;
			msgTerminate.root = rootNode->desc;
			msgTerminate.leftChild = rootNode->leftDesc;
			msgTerminate.rightChild = rootNode->rightDesc;
			msgTerminate.pLeft = reinterpret_cast<long int>(rootNode->leftChild);
			msgTerminate.pRight = reinterpret_cast<long int>(rootNode->rightChild);
			/*for(int i=0;i<numProcs;i++)
			{
				if(i!= procRank)
					GAL_SendMessage(i,MESSAGE_DONEVPTREE,&msgTerminate); 
			}*/
			done = true;
		}

	}
	/*else
	{
		//just create and leave it empty. Will be populated when MESSAGE_DONETREE is received.
		rootNode = new GAL_Vertex();
	}*/
	
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
				     	receive_oob(pg, msg.first, msg.second, msgBuildSubTree);
				GAL_Vertex node;
				GAL_Vertex* tmpLeaf;
#ifdef TRAVERSAL_PROFILE
					numLinks++;
#endif
				node.desc = msgBuildSubTree.subroot;
				node.parent = reinterpret_cast<GAL_Vertex*>(msgBuildSubTree.pLeaf);
				
				if((node.desc).owner !=  procRank)
					tmpLeaf = &node;
				else
				{
					tmpLeaf = reinterpret_cast<GAL_Vertex*>(msgBuildSubTree.pLeaf);
				}

				if(msgBuildSubTree.depth == subtreeHeight)
				{
					numUpdatesRequired+=(numProcs-2);
					donebuildsubtree = GAL_BuildSubTree(points,tmpLeaf, msgBuildSubTree.from, msgBuildSubTree.to, msgBuildSubTree.depth, 0, msgBuildSubTree.isleft);
				}
				else
					donebuildsubtree = GAL_BuildSubTree(msgBuildSubTree.ptv,tmpLeaf, 0, msgBuildSubTree.ptv.size(), msgBuildSubTree.depth, 0, msgBuildSubTree.isleft);
					
				//printf("process %d from:%d to:%d\n",procRank,msgBuildSubTree.from,msgBuildSubTree.to);
				

				//If all the vertices of the subtree are owned by a single process, inform the caller process that subtree construction is done.
				if(donebuildsubtree == BUILDSUBTREE_SAMEPROCESS)
				{
					//construct donesubtree message to notify parent

					msgDoneTree.parent = msgBuildSubTree.subroot.owner;
					if(msgBuildSubTree.isleft)
					{
						msgDoneTree.pRoot = reinterpret_cast<long int>(tmpLeaf->leftChild);
					}
					else
					{
						msgDoneTree.pRoot = reinterpret_cast<long int>(tmpLeaf->rightChild);
					}
			
					msgDoneTree.pLeaf = msgBuildSubTree.pLeaf;
					msgDoneTree.isLeft = msgBuildSubTree.isleft;

					//printf("Entire subtree belongs to single process.\n");
					GAL_SendMessage(msg.first,MESSAGE_DONESUBTREE, &msgDoneTree);
					
				}
				}
				break;
			case MESSAGE_DONESUBTREE:
				{
		              	receive_oob(pg, msg.first, msg.second, msgDoneTree);
				//Receive the node which is pseudo leaf.  A subtree of this node is fully constructed.
				GAL_Vertex* vert = reinterpret_cast<GAL_Vertex*>(msgDoneTree.pLeaf);
				GAL_Vertex* tmpVert = vert;
			
				//update the update count.
				(*g)[vert->desc].uCount++;
					
				if((*g)[vert->desc].uCount > 2)
					printf("NIKHIL ERRORCOUNT\n");

				//update the left or right child pointers
				if(msgDoneTree.isLeft)
				{
					vert->leftChild = reinterpret_cast<GAL_Vertex*>(msgDoneTree.pRoot);	
					vert->leftDesc = msg.first;
				}
				else
				{
					vert->rightChild = reinterpret_cast<GAL_Vertex*>(msgDoneTree.pRoot);	
					vert->rightDesc = msg.first;
				}
	
				//repeatedly update all the nodes up the tree if update count of any of those is 2.
				if((*g)[tmpVert->desc].uCount == 2)
				{
					
					while(tmpVert)
					{
						if(tmpVert->level == (subtreeHeight-1))
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
							msgUpdatePLeaf.label = tmpVert->label;
							vPLeaves.push_back(msgUpdatePLeaf);						
						}
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
																		//update parent's update count.
									(*g)[tmpVert->desc].uCount++;
									if((*g)[vert->desc].uCount > 2)
										printf("NIKHIL ERRORCOUNT\n");
									//if parent node has an update count of two, move up the tree.
									if((*g)[tmpVert->desc].uCount == 2)
										continue;
									else
										break;	
								}
								else
								{
									//Send message to the owning process.
									msgDoneTree.isLeft = tmpVert->isLeftChild;
									msgDoneTree.parent = tmpVert->parentDesc;
									msgDoneTree.pRoot = reinterpret_cast<long int>(tmpVert);
									msgDoneTree.pLeaf = reinterpret_cast<long int>(tmpVert->parent);
									GAL_SendMessage((tmpVert->parentDesc),MESSAGE_DONESUBTREE,&msgDoneTree);
									break;	
								}
						}
					}		
				}

				if(done)
				{		
					assert(procRank == 0);
					MsgUpdatePLeaves msgUpdatePLeaves;
					msgUpdatePLeaves.vPLeaves.insert(msgUpdatePLeaves.vPLeaves.begin(),vPLeaves.begin(), vPLeaves.end());
					//printf("%d NUM_PSEUDOLEAVES %d (%d)\n",procRank,vPLeaves.size(), pseudoLeaves.size());
					for(int i=1;i<numProcs;i++)
					{
						GAL_SendMessage(i,MESSAGE_UPDATE_PLEAVES,&msgUpdatePLeaves); 
					}	
					vPLeaves.erase(vPLeaves.begin(),vPLeaves.end());

				}
				}
				break;
			case MESSAGE_DONEVPTREE:
				{
				MsgTerminate msgTerminate;
		              	receive_oob(pg, msg.first, msg.second, msgTerminate);
				done = true;
				}
				break;
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
					pRoot->parents.push_back(std::make_pair(msg.first,msgUpdatePRoot.pLeaf));
					numUpdatesRequired--;
					//printf("process:%d updates:%d\n",process_id(g->process_group()), numUpdatesRequired);
					if(GAL_Aux_IsReadyToExit())
						done = true;
				}
				break;
			default:break;
		}
		
	}
		
	pseudoLeaves.erase(pseudoLeaves.begin(),pseudoLeaves.end());
#ifdef TRAVERSAL_PROFILE
	local_subgraph<BGLGraph> local_g2(*g);
	vertices(local_g2);
	if (num_vertices(local_g2) > 0) 
	{
		printf("Num of vertices in process : %d = %ld\n",procRank,num_vertices(local_g2)); 
	}

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
		printf("Total PRootLeaves:%ld\n",totalPRootLeaves);
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
	/*int localMaxHeight=0, maxHeightOfLeaf=0;
	std::map<int,int>::iterator it;
	for (it=numNodesAtHeight.begin(); it!=numNodesAtHeight.end(); ++it)
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
	 int *localLeavesAtHt = new int[maxHeightOfLeaf+1];
	 int *leavesAtHt = new  int[maxHeightOfLeaf+1];
	//long in t totalLeaves=0;
	for(int i=0;i<maxHeightOfLeaf+1;i++)
	{
		localLeavesAtHt[i]=0;
		leavesAtHt[i]=0;
	}
	for (it=numNodesAtHeight.begin(); it!=numNodesAtHeight.end(); ++it)
		localLeavesAtHt[it->first] = it->second;
	for(int i=0;i<maxHeightOfLeaf+1;i++)
	{
		reduce(communicator(pg),localLeavesAtHt[i], leavesAtHt[i], std::plus<int>(),0);
		if((procRank == 0) && (leavesAtHt[i] > 0))
		{
			printf("Total Nodes At Height: %d: %d\n",i,leavesAtHt[i]);
		}
	}
	delete [] leavesAtHt;
	delete [] localLeavesAtHt;
	numNodesAtHeight.clear();*/


#endif

	
return 0;
}

/*IMP: Caller of this function must delete the vertex type returned after its usage */

void* GAL::GAL_GetVertexProperties(GAL_Vertex& v, int vertextype)
{
#ifdef PARALLEL_BGL
	BGLTreeNode *ret = NULL;
	switch(vertextype)
	{
		case VPTREE: 
			ret = new BGLTreeNode();
			ret->uCount = (*g)[v.desc].uCount; 				
			break;
		default: break;

	}
	return ret;
#endif
}

void GAL::GAL_SetVertexProperties(GAL_Vertex& v, void* refnode, int vertextype)
{
#ifdef PARALLEL_BGL
	switch(vertextype)
	{
		case VPTREE: 
			(*g)[v.desc].uCount = reinterpret_cast<BGLTreeNode*>(refnode)->uCount;
			break;
	}
#endif
}

int GAL::GAL_BuildSubTree(TPointVector& points, GAL_Vertex* subtreeRoot,int from, int to, int depth, int DOR, bool isLeft)
{
	if(to <= from)
	{
		return BUILDSUBTREE_SAMEPROCESS;
	}
	//If the subtree of configured depth is created, ask a different process to take over the creation of subtrees that are down the hierarchy
#ifdef MERGE_HEIGHT1_TREES
	else if((numProcs> 1) && (DOR == subtreeHeight) && ((to - from) != 1))
#else
	else if((numProcs > 1) && (DOR == subtreeHeight))
#endif
	{
		pseudoLeaves.push_back(subtreeRoot);
		subtreeRoot->pseudoLeaf = true;
		if((procRank != 0) && (depth == subtreeHeight))
			return BUILDSUBTREE_MAXHEIGHT;
		MsgBuildSubTree m(from,to,isLeft,depth,subtreeRoot->desc);
		m.pLeaf = reinterpret_cast<long int>(subtreeRoot);
		if(depth!=subtreeHeight)
		{
			for(int i=from;i<to;i++)
			{
				m.ptv.push_back(points[i]);
			}	
		}
		//get next process
		int nextprocessid = GALHelper_GetNextProcessId(m.pLeaf);
		assert(nextprocessid != procRank);
		/*((procRank)* 2 + 1 + procCount) % numProcs;
		procCount++;
		if(procCount > (1<<subtreeHeight))
			procCount = 1;
		//printf("%d finished creating subtree of 2. sending build tree message to process %d, from:%d to:%d, curheight:%d\n",procRank,nextprocessid,from, to, depth);
		if(nextprocessid == procRank)
		{
			nextprocessid +=1;
			if(nextprocessid == numProcs)
				nextprocessid = 0;
		}
		if(nextprocessid == procRank)
			printf("FATAL ERROR2:%d\n",nextprocessid);*/
		GAL_SendMessage(nextprocessid,MESSAGE_BUILDSUBTREE,&m); 
		return BUILDSUBTREE_MAXHEIGHT;
	}
	else if((to - from) == 1)
	{
		//printf("reached leaf node\n");
		//Create leaf vertex and set properties
		//printf("leafnode. process %d from:%d to:%d (%f %f )\n",procRank,from,to,points.at(from).pt[0],points.at(from).pt[1]);
		GAL_Vertex* leafNode;
		if((DOR == 0) || (DOR == subtreeHeight))
		{
			leafNode = GAL_CreateVertex(VTYPE_PSEUDOROOT);
			leafNode->pseudoRoot = true;
		}
		else	
			leafNode = GAL_CreateVertex(VTYPE_NORMAL);
    		BGLTreeNode n;
		n.uCount = 2;
		GAL_SetVertexProperties(*leafNode, &n, VPTREE);
		leafNode->level = depth;
		leafNode->leaf = true;
		leafNode->point = points[from];
		//printf("%d: (%f %f)\n",leafNode->label, leafNode->point.pt[0],leafNode->point.pt[1]);
		leafNode->parent = subtreeRoot;
		leafNode->parentDesc = subtreeRoot->desc.owner;
		leafNode->isLeftChild = isLeft;
		std::pair<std::map<int,int>::iterator,bool> ret;
		ret = numNodesAtHeight.insert(std::pair<int,int>(depth,1) );
		if(ret.second==false)
			numNodesAtHeight[depth] += 1;

#ifdef TRAVERSAL_PROFILE
		std::pair<std::map<int,long int>::iterator,bool> ret1;
		ret1 = numLeavesAtHeight.insert(std::pair<int,long int>(depth,1) );
		if(ret1.second==false)
			numLeavesAtHeight[depth] += 1;
#endif
		//If parent belongs to a different node, Send message to parent that subtree construction is done.
		if((subtreeRoot->desc).owner != (leafNode->desc).owner)
		{
			leafNode->parentDesc = subtreeRoot->desc.owner;
			leafNode->parent = subtreeRoot->parent;

			MsgUpdateMinMax msgDoneTree;
			msgDoneTree.parent = subtreeRoot->desc.owner;
			msgDoneTree.pRoot = reinterpret_cast<long int>(leafNode);
			msgDoneTree.pLeaf = reinterpret_cast<long int>(subtreeRoot->parent);
			msgDoneTree.isLeft = isLeft;
			GAL_SendMessage((subtreeRoot->desc).owner,MESSAGE_DONESUBTREE,&msgDoneTree);
#ifdef TRAVERSAL_PROFILE
			numPRootLeaves++;
#endif

			return BUILDSUBTREE_SENTOOB;
		}
		else
		{
			if(isLeft)
			{
				subtreeRoot->leftChild = leafNode;
				subtreeRoot->leftDesc = procRank;
			}
			else
			{
				subtreeRoot->rightChild = leafNode;
				subtreeRoot->rightDesc = procRank;
			}

			//If parent belongs to same node, Update parent's bounds.
			return BUILDSUBTREE_SAMEPROCESS;
		}
	}
	else
	{
		//reached int node 
		int oobsent = BUILDSUBTREE_SAMEPROCESS;
		int flag =BUILDSUBTREE_SAMEPROCESS;

		//int i = (int)((float)rand() / RAND_MAX * (to - from - 1) ) + from;
		int i=from;
		std::swap(points[from], points[i]);
		int median = ( from + to) / 2;
		// partition around the median distance
		std::nth_element(
				points.begin() + from + 1,
				points.begin() + median,
				points.begin() + to,
				DistanceComparator(points[from]));
		// what was the median?

	//printf("%d: intnode before sorted(%d %d) based on depth:%d, %f %f %f %f %f %f %f\n",procRank,from,to,depth,points.at(0).pt[0],points.at(1).pt[0],points.at(2).pt[0],points.at(3).pt[0],points.at(4).pt[0],points.at(5).pt[0],points.at(6).pt[0]);
		//create a vertex for the intermediate node and add it to the graph
		GAL_Vertex* intNode; 
		if((DOR == 0)||(DOR==subtreeHeight))
		//if(((DOR == 0)||(DOR==13)) && (subtreeRoot!=NULL))
		{
			intNode = GAL_CreateVertex(VTYPE_PSEUDOROOT);
		}
		else	
			intNode = GAL_CreateVertex(VTYPE_NORMAL);

		intNode->threshold = mydistance((points[from]), (points[median]));
		intNode->level = depth;
		intNode->point = points[from];
		//printf("%d: (%f %f)\n",intNode->label, intNode->point.pt[0],intNode->point.pt[1]);
		intNode->isLeftChild = isLeft;
		std::pair<std::map<int,int>::iterator,bool> ret;
		ret = numNodesAtHeight.insert(std::pair<int,int>(depth,1) );
		if(ret.second==false)
			numNodesAtHeight[depth] += 1;

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
				intNode->pseudoRoot = true;
			}
			
			//update parents left and right children pointers and descriptors
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
			rootNode = intNode;
			rootNode->parent = NULL;
		}
	

		//build left tree
		oobsent = GAL_BuildSubTree(points,intNode,from+1, median,depth+1, DOR+1, true);

		//We need to update the parent of the subtree rooted at intNode that left subtree construction is done or not done.
		if(oobsent != BUILDSUBTREE_FAILURE)
		{
			//if((oobsent == BUILDSUBTREE_SAMEPROCESS) && ((subtreeRoot->desc).owner == (intNode->desc).owner))
			if((oobsent == BUILDSUBTREE_SAMEPROCESS))
			{
				//update intNode bounding box
				BGLTreeNode *node = (BGLTreeNode *) (GAL_GetVertexProperties(*intNode, VPTREE));
				node->uCount++;
				GAL_SetVertexProperties(*intNode,node,VPTREE);
				delete node;
			}
			else
			{
				if(oobsent != BUILDSUBTREE_MAXHEIGHT)
				{
					flag = BUILDSUBTREE_SENTOOB;
				}
				else
				{
					flag = oobsent;
				}
			}

		}

	//printf("%d: intnode after leftsorted(%d %d) based on depth:%d , %f %f %f %f %f %f %f\n",procRank,from,to,depth,points.at(0).pt[0],points.at(1).pt[0],points.at(2).pt[0],points.at(3).pt[0],points.at(4).pt[0],points.at(5).pt[0],points.at(6).pt[0]);
		//build right tree
		oobsent = GAL_BuildSubTree(points,intNode, median, to,depth+1, DOR+1, false);
		
		if(oobsent != BUILDSUBTREE_FAILURE)
		{
			//if((oobsent == BUILDSUBTREE_SAMEPROCESS) && ((subtreeRoot->desc).owner == (intNode->desc).owner))
			if((oobsent == BUILDSUBTREE_SAMEPROCESS))
			{
					//update intNode bounding box
				//update count property is to be incremented. To modify update count, first, fetch the existing vertex properties and then update.
				BGLTreeNode* node = (BGLTreeNode *) (GAL_GetVertexProperties(*intNode, VPTREE));
				node->uCount++;
				GAL_SetVertexProperties(*intNode,node,VPTREE);
				delete node;
			}
			else
			{
				if(oobsent != BUILDSUBTREE_MAXHEIGHT)
				{
					flag = BUILDSUBTREE_SENTOOB;
				}
				else
				{
					if(flag == BUILDSUBTREE_SENTOOB)
						printf("FATAL ERROR:%d",oobsent);
					flag = oobsent;
				}

			}

		}
		if(flag == BUILDSUBTREE_SENTOOB)
		{
				if(subtreeRoot && (subtreeRoot->desc).owner != (intNode->desc).owner)
				{
					MsgUpdateMinMax msgDoneTree;
					msgDoneTree.parent = subtreeRoot->desc.owner;
					msgDoneTree.pRoot = reinterpret_cast<long int>(intNode);
					msgDoneTree.pLeaf = reinterpret_cast<long int>(subtreeRoot->parent);
					msgDoneTree.isLeft = isLeft;
					GAL_SendMessage((subtreeRoot->desc).owner,MESSAGE_DONESUBTREE,&msgDoneTree);
					flag = BUILDSUBTREE_SENTOOB;
				}
	
		}
		
		return flag;

	}
	
}

GAL_Edge GAL::GAL_CreateEdge(GAL_Vertex& source, GAL_Vertex& target)
{
#ifdef PARALLEL_BGL
	GAL_Edge edge;
	std::pair<BGLEdgedesc,bool> p = add_edge(source.desc, target.desc,*g);
	
	edge.desc = p.first;
	return edge;
#endif
}

void GAL::GAL_SetEdgeProperties(GAL_Edge& edge, void* refedge, int vertexType)
{
#ifdef PARALLEL_BGL
	switch(vertextype)
	{
		case VPTREE: (*g)[edge.desc].left = reinterpret_cast<BGLTreeEdge*>(refedge)->left;
			break;
		default:break;
	}
#endif

}

void* GAL::GAL_GetEdgeProperties(GAL_Edge& e, int vertextype)
{
#ifdef PARALLEL_BGL
	BGLTreeEdge *ret = NULL;
	switch(vertextype)
	{
		case VPTREE: 
			ret = new BGLTreeEdge();
			ret->left = (*g)[e.desc].left; 				
			break;
		default: break;

	}
	return ret;
#endif
}

GAL_Vertex* GAL::GAL_GetStartVertex()
{
	return rootNode;
}

void GAL::GAL_PrintGraph()
{

#ifdef PARALLEL_BGL
#ifdef TRAVERSAL_PROFILE
	if(procRank == 0)
	{
		std::vector<long int>::iterator iter = pipelineStage.begin();
		for(;iter!=pipelineStage.end();iter++)
		{
			GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(*iter);
			if((pRoot->level != 0) && (pRoot->leftChild || pRoot->rightChild))
			{
				long int count=0;
				GALHelper_CountSubtreeNodes(pRoot, count);
				//printf("Subtree %ld Points_Visited %ld\n",*iter,count);
			}
			
		}
	}
	long int count=0;
	GALHelper_CountSubtreeNodes(rootNode, count);
	printf("Subtree %ld Points_Visited %ld\n",reinterpret_cast<long int>(rootNode),count);
	double bof=0.;
	GALHelper_GetBlockOccupancyFactor(rootNode, bof);
	printf("Bof:  %f\n",bof/count);

#endif

#ifdef STATISTICS
	long int totalTraverseCount;
	reduce(communicator(pg),traverseCount, totalTraverseCount, std::plus<long int>(),0);
	float localPPM=0., avgPPM;
	if(pointsProcessed != 0)
		localPPM = pointsProcessed/(float)traverseCount;
		
	reduce(communicator(pg),localPPM, avgPPM, std::plus<float>(),0);
	if(procRank==0)
		printf("Total Messages Processed:%d Avergage Points Per Message:%f\n",totalTraverseCount, avgPPM/numProcs);

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
	}
#endif
#endif
#endif	
}

int GAL::GALHelper_HandleMessageTraverseBackward(GAL_Vertex* childNode, GALVisitor* vis, TBlockId curBlockId)
{
	int ret = STATUS_TRAVERSE_INCOMPLETE;
	//GAL_Vertex* tmpChildNode = const_cast<GAL_Vertex *>(childNode);
	GAL_Vertex* tmpChildNode = childNode;
	GAL_Vertex* nextNodeToVisit = NULL;
#ifdef PERFCOUNTERS
	assert(compTimer.IsRunning());
#endif

	while(true)
	{
		TBlockId fragBlkId;
		fragBlkId.second = INVALID_BLOCK_ID; 
		BlockStackListIter curBStack = vis->GALVisitor_GetCurrentBlockStack(); 
		bool readyToSend = vis->GALVisitor_IsLastFragment(tmpChildNode, curBStack, fragBlkId);	
		if(fragBlkId.second != INVALID_BLOCK_ID)
		{
			vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
			vis->GALVisitor_RemoveBlockStack();
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
						}
					}
				}
#endif
				vis->GALVisitor_SetAsCurrentBlockStack(curBlockId);
			}
		}
			
		long int parent = reinterpret_cast<long int>(tmpChildNode->parent);
		int procId = (tmpChildNode->parentDesc);
		if(tmpChildNode->pseudoRoot && (tmpChildNode->level == subtreeHeight))
		{
			curBStack = reinterpret_cast<BlockStack*>(curBlockId.second);
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
			msgTraverse.pLeafThreshold = ((GAL_PseudoRoot*)(tmpChildNode))->parentThreshold;
			GALHelper_GetNodeData(tmpChildNode,msgTraverse);
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
					GALHelper_SetNodeData(tmpNextNodeToVisit,msgTraverse);
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
					msgTraverse.pLeafThreshold = prnt->threshold;//((GAL_PseudoRoot*)(tmpChildNode))->parentThreshold;
					GALHelper_GetNodeData(tmpChildNode,msgTraverse);
					//vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
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
			ret = GAL_TraverseHelper(vis,nextNodeToVisit, NULL);
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
	bool found = false;
	long int origBlkId;
	assert(!compTimer.IsRunning());
	compTimer.Start();
#endif

	assert(vis);
	//if(procRank == 0)
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
#ifdef PERFCOUNTERS
						compTimer.Stop();
#endif
						readyToFlush = true;
						int sentVal = 0;
						for(int i=1;i<numProcs;i++)
						{
							GAL_SendMessage(i,MESSAGE_SENDCOMPRESSED,&sentVal); 
						}
					}
#endif
				}

			}
			else
			{
				int ret = GAL_TraverseHelper(vis,rootNode, NULL);
				
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
	{
		if(aggrBuffer.size() > 0)
		{
			GAL_TraverseHelper_SendCompressedMessages(vis);
		}
	}
#endif

	return ret;
}




int GAL::GAL_Traverse(GALVisitor* vis, int blkSize, int nBuffers, std::vector<int> pBufferSizes)
{
	PBGL_oobMessage msg;
	bool done = false;
	int ret = STATUS_TRAVERSE_INCOMPLETE;
	blockSize = blkSize;

#ifdef PERFCOUNTERS
	timer totTimer;
	totTimer.Start();
	compTimer.Start();
	workloadTimer.Start();
#endif	
#ifdef MESSAGE_AGGREGATION
	numPipelineBuffers = nBuffers;	
	assert((numPipelineBuffers > 0) && (numPipelineBuffers <= 3));
	pipelineBufferSize = new int[numPipelineBuffers];
	for(int i=0;i<numPipelineBuffers;i++)
	{
		if(i == 0)
			pipelineBufferSize[i] = PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,1);
		else if (i == 1)
			pipelineBufferSize[i] = PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,2);
		else if (i == 2)
			pipelineBufferSize[i] = PIPELINE_BUFFER_SIZE(PIPELINE_BUFFER_SIZE_LEVEL_,3);
	}
	if(pBufferSizes.size() > 0)
	{
		for(int i=0;i<pBufferSizes.size();i++)
		{
			pipelineBufferSize[i] = pBufferSizes[i];
		}
	}
#ifdef STATISTICS
	bufferStage = new long int[numPipelineBuffers+1];
	for(int i=0;i<(numPipelineBuffers+1);i++)
		bufferStage[i]=0;
#endif
#endif

#ifdef PERFCOUNTERS
	double lastTime = clock();
	compTimer.Stop();
#endif
	while (!done)
	{
#ifdef PERFCOUNTERS
		if((clock() - lastTime) >= timeStep)
		{
			BlockStats bs = vis->GALVisitor_GetNumberOfBlocksInFlight();
			tpFinishedBlocks.push_back(bs.numUniqBlocks);
			lastTime = clock();
		}
#endif

		//poll for messages
		PBGL_AsyncMsg pollMsg = pg.poll();
		if(!pollMsg)
		{
			//if(procRank == 0)
			{
				if(vis)
				{
					ret = GAL_TraverseHelper_SendBlocks(vis);
					if(ret == STATUS_TRAVERSE_COMPLETE)
					{
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
										GAL_SendMessage(i,MESSAGE_DONEVPTREE,&msgTerminate); 
								}
								break;
							}
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
						GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(msgTraverse.pRoot);
						GAL_PseudoRoot* pSibling = reinterpret_cast<GAL_PseudoRoot*>(msgTraverse.pSibling);
#ifdef PERFCOUNTERS
						compTimer.Start();
#endif
#ifdef STATISTICS
						traverseCount++;
						pointsProcessed += msgTraverse.l.size();
#endif
						((GAL_PseudoRoot*)(pRoot))->parentThreshold = msgTraverse.pLeafThreshold;
						GALHelper_SetNodeData(pRoot,msgTraverse);
						/*if(pSibling)
						{
							((GAL_PseudoRoot *)(pRoot))->pSibling = pSibling;
							((GAL_PseudoRoot *)(pRoot))->siblingDesc = msgTraverse.siblingDesc;
						}*/

						/*if(msgTraverse.blkStart.second == 67108878)
						{
							printf("Debug 9 procRank:%d Sender:%d\n",procRank, msg.first);
						}*/
						int procId =(pRoot->parentDesc);
						bool loadBalanced = false;
						if(pRoot->level == subtreeHeight)
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

						ret = GALHelper_DeAggregateBlockAndSend(vis, &pRoot, msgTraverse, loadBalanced);

						if(ret == STATUS_TRAVERSE_COMPLETE)
						{
							if(!loadBalanced)
								int re = vis->GALVisitor_RemoveBlockStack();	

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
									}
								}
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
								long int origBlkId = vis->GALVisitor_DetermineOrigBlkId(msgTraverse.l[0]);
								//std::map<long int, PipelineStats>::iterator blkIter = (iter->second).find(msgTraverse.blkStart.second);
								std::map<long int, PipelineStats>::iterator blkIter = (iter->second).find(origBlkId);
								assert(blkIter != (iter->second).end());
								(blkIter->second).timeInStage += stageExTime;
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
#ifdef STATISTICS
						//traverseCount++;
#endif
#ifdef PERFCOUNTERS
						compTimer.Start();
#endif
						/*if(msgBkTraverse.blkStart.second == 67108878)
						{
							printf("Debug 8 procRank:%d Sender:%d\n",procRank, msg.first);
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
								status = GALHelper_HandleMessageTraverseBackward_Multiple(parentNode, vis, blkIdSet);
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
								long int origBlkId = vis->GALVisitor_DetermineOrigBlkId(msgBkTraverse.l[0]);
								std::map<long int, PipelineStats>::iterator blkIter = (iter->second).find(origBlkId);
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
					readyToFlush = true;
					GAL_TraverseHelper_SendCompressedMessages(vis);
				}
				break;
#endif
				case MESSAGE_DONEVPTREE:
				{
					MsgTerminate msgTerminate;
						receive_oob(pg, msg.first, msg.second, msgTerminate);
					ret = msgTerminate.pLeft;
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
#ifdef MESSAGE_AGGREGATION
	if(numPipelineBuffers > 0)
	{
		if(pipelineBufferSize)
			delete [] pipelineBufferSize;
	}
#endif
#ifdef PERFCOUNTERS
	workloadTime = workloadTimer.GetTotTime();
	totTimer.Stop();
	totTime = totTimer.GetTotTime();
	compTime = compTimer.GetTotTime();
#endif	
	return ret;
}


/* Description: This is a recursive function called to traverse the tree in an algorithm specific manner.
 * Parameters:  vis - IN - reference to the visitor. 
 * 		node - IN - traverse the subtree rooted at this 'node'.
 * 		sib - IN - next node to be traversed if the traversal at this 'node' returns with a status STATUS_TRAVERSE_COMPLETE. This is null if there are no more subtrees to traverse after
 * 		successfully completing the traversal of this 'node'.
 * Return Value: status of the traversal:
 * 		-STATUS_TRAVERSAL_COMPLETE - if traversal is complete
 * 		-STATUS_TRAVERSAL_INCOMPLETE - if traversal proceeded to a subtree owned by a different process.
 * Precondtions: Visitor should take care that the 'block' - set of points that visit this 'node' is correctly set. Also, 'block ID' and the reference to current 'blockStackList' 
 * (only in case of simultaneous traversal of multiple blocks) should be correctly set by the visitor.
 * Postconditions: Updates to the block (Depending upon which points in the block decided to proceed further with the traversal and which got truncated at this 'node').
 * 		  -Updates to a point's locally computed data when it visits a node (number of nodes traversed, correlation, closest_dist etc)
 * 		  -The top of the the block stack contains references to points that visited this node.
 */

int GAL::GAL_TraverseHelper(GALVisitor* vis, GAL_Vertex* node, GAL_Vertex* sib)
{
		TIndices leftBlock, rightBlock, superBlock;
		int ret = STATUS_TRAVERSE_COMPLETE;
		TBlockId blockId;
		long int uniqBlkId1, uniqBlkId2;
		if(!node)
			return STATUS_TRAVERSE_COMPLETE;
		assert((node->desc).owner ==procRank); 
		BlockStackListIter rightBStack, curBStack = vis->GALVisitor_GetCurrentBlockStack();
		
		bool traverseSubTree = vis->GALVisitor_VisitNode(node, sib, leftBlock, rightBlock, blockId);
		if(!traverseSubTree)
		{
			//printf("%d\n",node->level);
			return STATUS_TRAVERSE_COMPLETE;
		}


		if(node->leaf)
		{
			return STATUS_TRAVERSE_COMPLETE;
		}
		
		
		int expectedUpdates = 2 * ((leftBlock.size()>0) + (rightBlock.size()>0));
		
		if(expectedUpdates > 2)
		{
			curAggrBlkId += 1;
			uniqBlkId1 = COMPOSE_BLOCK_ID(procRank,curAggrBlkId);
			curAggrBlkId += 1;
			uniqBlkId2 = COMPOSE_BLOCK_ID(procRank,curAggrBlkId);
			
			rightBStack = vis->GALVisitor_CreateBlockStack(rightBlock, node, NULL, 2, curBStack);
			vis->GALVisitor_CreateBlockStack(leftBlock, node, NULL, 2, curBStack);
			/*if(procRank == 2)
			{
			printf("c: pRoot: %ld, BlkId: %ld \n", reinterpret_cast<long int>(node),uniqBlkId1);
			printf("c: pRoot: %ld, BlkId: %ld \n", reinterpret_cast<long int>(node),uniqBlkId2);
			}*/

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
				msgTraverse.pLeafThreshold = node->threshold;
				GALHelper_GetNodeData(node,msgTraverse, false);
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
					aggrBlockSize = pipelineBufferSize[stageNum -1];
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
						msgTraverseRight.pLeafThreshold = node->threshold;
						GALHelper_GetNodeData(node,msgTraverseRight,false);
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
					//printf("(Process %d, node:%p) buffered %d bytes at buffer stage %d\n",procRank, node, bufCount, (*g)[node->desc].level);
					ret = STATUS_TRAVERSE_INCOMPLETE;
				}
#ifdef STATISTICS
				if(bufCount>=aggrBlockSize)
				{
					if(stageNum <= numPipelineBuffers)
						bufferStage[stageNum] += bufCount;
				}
#endif
#endif
			}
			else
			{
				ret = GAL_TraverseHelper(vis,node->leftChild, node->rightChild);
				if(ret == STATUS_TRAVERSE_COMPLETE)
				{
					//vis->GALVisitor_SetBlock(leftBlock);
					if(node->rightChild && ((node->rightDesc) != procRank))
					{	
#ifdef PERFCOUNTERS
						compTimer.Stop();
#endif
						MsgTraverse msgTraverse;
						blockId = vis->GALVisitor_GetLocalData(msgTraverse.l);
						msgTraverse.blkStart = blockId;
						msgTraverse.pRoot = reinterpret_cast<long int>(node->rightChild);
						GALHelper_GetNodeData(node,msgTraverse,false);
						msgTraverse.pLeafThreshold = node->threshold;
						msgTraverse.pLeaf = reinterpret_cast<long int>(node);
						msgTraverse.pSibling = static_cast<long int>(0);
						msgTraverse.siblingDesc = node->rightDesc;
						GAL_SendMessage((node->rightDesc),MESSAGE_TRAVERSE, &msgTraverse);
						ret = STATUS_TRAVERSE_INCOMPLETE;
					}
					else
					{
						ret = GAL_TraverseHelper(vis,node->rightChild, NULL);
						if(ret == STATUS_TRAVERSE_COMPLETE)			
						{
							vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
							if(expectedUpdates > 2)
							{
								int re = vis->GALVisitor_RemoveBlockStack();
								if(re == -1)
									printf("Debug 9\n");
								assert(curBStack->numFragments > 0);
							}
						}	
					}
				}
					
			}
				
			
		}
		

		if(expectedUpdates > 2)
		{
#ifdef PERFCOUNTERS
			if(!(compTimer.IsRunning()))
				compTimer.Start();
#endif
			assert(rightBlock.size() > 0);
			vis->GALVisitor_SetAsCurrentBlockStack2(rightBStack);
		}
		
		int ret2 = STATUS_TRAVERSE_COMPLETE;
		if(rightBlock.size() > 0)
		{
			rightBlock.erase(rightBlock.begin(),rightBlock.end());
			//first traverse right subtree
			if(node->rightChild && ((node->rightDesc) != procRank))
			{	
#ifdef PERFCOUNTERS
				compTimer.Stop();
#endif
				MsgTraverse msgTraverse, msgTraverseRight;
				blockId = vis->GALVisitor_GetLocalData(msgTraverseRight.l);
				msgTraverseRight.pRoot = reinterpret_cast<long int>(node->rightChild);
				GALHelper_GetNodeData(node,msgTraverseRight,false);
				msgTraverseRight.pLeafThreshold = node->threshold;
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
					aggrBlockSize = pipelineBufferSize[stageNum -1];
					bufCount = GAL_TraverseHelper_CompressMessages(vis, msgTraverse, aggrBlockSize, msgTraverseRight,false);
				}

				if(bufCount >= aggrBlockSize)
				{
#endif
				GAL_SendMessage((node->rightDesc),MESSAGE_TRAVERSE, &msgTraverseRight);
				ret2 = STATUS_TRAVERSE_INCOMPLETE;
#ifdef MESSAGE_AGGREGATION
					if(msgTraverse.l.size() > 0)
					{
						GALHelper_GetNodeData(node,msgTraverse,false);
						msgTraverse.pLeafThreshold = node->threshold;
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
						GAL_SendMessage((node->leftDesc),MESSAGE_TRAVERSE, &msgTraverse);
						ret = STATUS_TRAVERSE_INCOMPLETE;
					}
				}
				else
				{
					//printf("(Process %d, node:%p) buffered %d bytes at buffer stage %d\n",procRank, node, bufCount, (*g)[node->desc].level);
					ret2 = STATUS_TRAVERSE_INCOMPLETE;
				}
#ifdef STATISTICS
				if(bufCount>=aggrBlockSize)
				{
					if(stageNum <= numPipelineBuffers)
						bufferStage[stageNum] += bufCount;
				}
#endif
#endif
			}
			else
			{
				ret2 = GAL_TraverseHelper(vis,node->rightChild, node->leftChild);
				if(ret2 == STATUS_TRAVERSE_COMPLETE)
				{
					//vis->GALVisitor_SetBlock(rightBlock);
					if(node->leftChild && ((node->leftDesc) != procRank))
					{	
#ifdef PERFCOUNTERS
						compTimer.Stop();
#endif
						MsgTraverse msgTraverse;
						blockId = vis->GALVisitor_GetLocalData(msgTraverse.l);
						msgTraverse.blkStart = blockId;
						msgTraverse.pRoot = reinterpret_cast<long int>(node->leftChild);
						GALHelper_GetNodeData(node,msgTraverse,false);
						msgTraverse.pLeafThreshold = node->threshold;
						msgTraverse.pLeaf = reinterpret_cast<long int>(node);
						msgTraverse.pSibling = static_cast<long int>(0);
						msgTraverse.siblingDesc = node->leftDesc;
						GAL_SendMessage((node->leftDesc),MESSAGE_TRAVERSE, &msgTraverse);
						ret2 = STATUS_TRAVERSE_INCOMPLETE;
					}
					else
					{
						ret2 = GAL_TraverseHelper(vis,node->leftChild, NULL);
						if(ret2 == STATUS_TRAVERSE_COMPLETE)			
						{
							vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
							if(expectedUpdates > 2)
							{
								int re = vis->GALVisitor_RemoveBlockStack();
								if(re == -1)
									printf("Debug 10\n");
								vis->GALVisitor_SetAsCurrentBlockStack2(curBStack);
							}
						}
					}
				}
					
			}
				
		}

		if((ret2 == STATUS_TRAVERSE_COMPLETE) && (ret == STATUS_TRAVERSE_COMPLETE))		
		{
			if(expectedUpdates > 2)
				vis->GALVisitor_PopFromCurrentBlockStackAndUpdate();
			ret  = STATUS_TRAVERSE_COMPLETE;
		}
		else
		{
#ifdef PERFCOUNTERS
			if(compTimer.IsRunning())
				compTimer.Stop();
#endif
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
int GAL::GAL_TraverseHelper_CompressMessages(GALVisitor* vis,MsgTraverse& msgTraverse, int aggrBlockSize, MsgTraverse& msgTraverseRight, bool goingLeft)
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
				GALHelper_GetNodeData(pLeaf,msgTraverse,false);
				msgTraverse.pLeafThreshold = pLeaf->threshold;
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
				GALHelper_GetNodeData(pLeaf,msgTraverseRight,false);
				msgTraverseRight.pLeafThreshold = pLeaf->threshold;
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
			ret = GAL_TraverseHelper(vis,tmpVertex, NULL);
			
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
					GALHelper_SetNodeData(pSibling,msgTraverse);
					int nextNodeProc = 0;
					std::list<BlockStack*>::iterator fIter = tmpFTable.begin();
					for(;fIter != (tmpFTable).end();fIter++)
					{
						vis->GALVisitor_SetAsCurrentBlockStack2(*fIter);
						ret = GAL_TraverseHelper(vis, pSibling, NULL);
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
			GALHelper_GetNodeData(tmpVertex,tmp);
			tmp.pLeafThreshold = ((GAL_PseudoRoot*)(tmpVertex))->parentThreshold;
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
		ret = GAL_TraverseHelper(vis,*pRoot, pSibling);
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
					compTimer.Stop();
#endif
					//pLeafThreshold nd parentCoords remain the same.
					//sending entire compressed data to sibling
					msgTraverse.pLeaf = msgTraverse.pLeaf;
					msgTraverse.pRoot = msgTraverse.pSibling;
					msgTraverse.pSibling = static_cast<long int>(0);
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
					GALHelper_SetNodeData(*pRoot,msgTraverse);
					if(!loadBalanced)
					vis->GALVisitor_UpdateCurrentBlockStackId(*pRoot);
					ret = GAL_TraverseHelper(vis,*pRoot,NULL);
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


void GAL::GALHelper_GetNodeData(GAL_Vertex* node, MsgTraverse& msg, bool moveBkwd)
{
	if(node->pseudoRoot && moveBkwd)
	{
		GAL_PseudoRoot* pRoot = (GAL_PseudoRoot*)(node);
		for(int i=0;i<DIMENSION;i++)
		{
			msg.pLeafCoords[i]=pRoot->parentCoord[i];
		}
	}
	else
	{
		for(int i=0;i<DIMENSION;i++)
		{
			msg.pLeafCoords[i]=(node->point).pt[i];
		}
	}
}

void GAL::GALHelper_SetNodeData(GAL_Vertex* node, const MsgTraverse& msg)
{
	if(node->pseudoRoot)
	{
		GAL_PseudoRoot* pRoot = (GAL_PseudoRoot*)(node);
		pRoot->parentThreshold = msg.pLeafThreshold;
		for(int i=0;i<DIMENSION;i++)
		{
			pRoot->parentCoord[i] = msg.pLeafCoords[i];
		}
	}
	else
	{
		assert(0);
		for(int i=0;i<DIMENSION;i++)
		{
			(node->point).pt[i] = msg.pLeafCoords[i];
		}
	}
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
			//if(((*leafTableIter)->desc).local == (msgIter->leafDesc).local)
			if((*leafTableIter)->label == msgIter->label)
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
						child->parentThreshold = (*leafTableIter)->threshold;
						for(int i=0;i<DIMENSION;i++)
							child->parentCoord[i] = (*leafTableIter)->point.pt[i];
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
						child->parentThreshold = (*leafTableIter)->threshold;
						for(int i=0;i<DIMENSION;i++)
							child->parentCoord[i] = (*leafTableIter)->point.pt[i];
					}
				}
				break;
			}
		}
	}
	
	return ret;
}

int GAL::GALHelper_GetNextProcessId(long int pLeaf)
{
	//get next process and send the message
	int nextprocessid = ((procRank)* 2 + 1 + procCount) % numProcs;
	procCount++;
	if(procCount > (1<<(subtreeHeight)))
	procCount = 1;
	if(nextprocessid == procRank)
	{
	nextprocessid +=1;
	if(nextprocessid == numProcs)
		nextprocessid = 0;
	}
	/*int nextprocessid;
	std::pair<std::map<long int, int>::iterator, bool> ret = pLeafMap.insert(std::make_pair(pLeaf,0));
	if(ret.second == false)
		return (ret.first)->second;
	else
	{
		while(1)
		{
			nextprocessid = rand() % (numProcs);
			if(nextprocessid != procRank)
				break;
		}
	}
	(ret.first)->second = nextprocessid;*/
	return nextprocessid;
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



void GAL::GALHelper_CountSubtreeNodes(GAL_Vertex* ver, long int& count)
{
		int ret = STATUS_TRAVERSE_COMPLETE;
#ifdef TRAVERSAL_PROFILE
		count += 1; //ver->pointsVisited;
#else
		count++;
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
				GALHelper_CountSubtreeNodes(ver->leftChild,count);
		}
		if(ver->rightChild)
		{
			if(ver->rightDesc == procRank)
				GALHelper_CountSubtreeNodes(ver->rightChild,count);
		}
			
		
}

#ifdef PERFCOUNTERS
GAL_Vertex* GAL::GALHelper_GetAncestorPseudoRoot(GAL_Vertex* node)
{
	GAL_Vertex* ret = node;
	while(ret->parent && ((ret->parentDesc) == procRank))
	{
		ret = ret->parent;	
	}
	if(!(ret->parent))
		return ret;
	return ret;
}
#endif

void GAL::GALHelper_CountReplicatedVertices(GAL_Vertex* node)
{
	replicatedVertexCounter++;
	if(node->leftChild && (node->leftDesc == procRank))
		GALHelper_CountReplicatedVertices(node->leftChild);
	if(node->rightChild && (node->rightDesc == procRank))
		GALHelper_CountReplicatedVertices(node->rightChild);
}

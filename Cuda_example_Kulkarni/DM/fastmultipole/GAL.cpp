/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "GAL.h"
#include "CorrelationVisitor.h"
#include "timer.h"
#include "AuxFn.h"
#include<boost/graph/distributed/depth_first_search.hpp>
#include<boost/bind.hpp>
#ifdef PAPI
#include"papi.h"
void handle_error(int retval)
{
	printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	exit(1);
}
#endif

int depth=0;
std::vector<GAL_Vertex*> allLeaves;
long int numExpectedUpdates;
//std::map<int,std::vector<int> > nodeSeenAtChkPt;
#ifdef SPAD_2
extern int numReplicatedSubtrees;
int numAdjacentSubtrees;
long int numReplicatedVertices;
#endif

typedef struct PLeafPointBucket{
int depth;
long int pLeaf;
long int from;
long int to;
long int totalPts;
}PLeafPointBucket;

std::vector<std::pair<int, PLeafPointBucket> > pLeafPointsPerProcess;

MsgUpdatePLeaves msgUpdatePLeaves;
long int labelCount; //counter to assign label values to a node.
std::map<long int,int> pLeafMap;
std::vector<MsgUpdatePLeaf> vPLeaves;
#ifdef TRAVERSAL_PROFILE
std::vector<long int> pipelineStage;
#endif

typedef optional<PBGL_oobMessage> PBGL_AsyncMsg;
#ifdef PERFCOUNTERS
timer compTimer;
timer leafStageTimer;
timer workloadTimer;
uint64_t workloadTime;
#endif

bool SortIntPairsBySecond(const std::pair<int, int>& a, const std::pair<int, int>& b)
{
	return a.second < b.second;
}

bool SortLongIntPairsBySecond_Decrease(const std::pair<long int, long int>& a, const std::pair<long int, long int>& b)
{
	return a.second >= b.second;
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
	return *g;
}

GAL_Vertex* GAL::GAL_GetRootNode()
{
	return rootNode;
}

GAL_Vertex* GAL::GAL_CreateVertex(TVertexType vType)
{
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
}

void GAL::GAL_SendMessage(int processId, int messageId, void* msg)
{
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
#ifdef MESSAGE_AGGREGATION
		case MESSAGE_SENDCOMPRESSED:
					send_oob(pg,processId,messageId,*(reinterpret_cast<int*>(msg))); 
					break;
#endif
		case MESSAGE_UPDATE_PLEAVES:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgUpdatePLeaves*>(msg))); 
					break;
		case MESSAGE_READYTOEXIT:
					send_oob(pg,processId,messageId,*(reinterpret_cast<int*>(msg))); 
					break;
#ifdef SPAD_2
		case MESSAGE_REPLICATE_REQ:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgReplicateReq*>(msg))); 
					break;
		case MESSAGE_REPLICATE_SUBTREE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgReplicateSubtree*>(msg))); 
					break;
#endif
		case MESSAGE_CREATE_SUBTREE:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgReplicateSubtree*>(msg))); 
					break;
		default:
			break;
	}
}

int GAL::GAL_ConstructQuadTree(Point* points, Box& boundingBox, int numPoints, bool hasMoreData, int stHeight)
{
	bool done = false;
	subtreeHeight = stHeight;
	int curLeafId;
	long int curPointIndx=0, prevPointIndx=-1;
	bool replicatedTopSubtree=false;
	bool curBucketSent = false;

	MsgBuildSubTree msgSubtreeSecondLevel;
	bool receivedDataInBatches = false;
	
	int donebuildsubtree;

	bool flag = true;
	if(!rootNode)
		rootNode = GAL_CreateVertex(VTYPE_PSEUDOROOT);
	for(long int i = 0;i<numPoints;i++)
	{
		rootNode->mass += points[i].mass;
		(rootNode->myPoints).push_back(points[i]);
	}
	
	/*if(ex > rootNode->box.endX)
		rootNode->box.endX = ex;
	else
		ex = rootNode->box.endX;

	if(ey > rootNode->box.endY)
		rootNode->box.endY = ey;
	else	
		ey = rootNode->box.endY;
		
	if(sx < rootNode->box.startX)
		rootNode->box.startX = sx;
	else
		sx = rootNode->box.startX;

	if(sy < rootNode->box.startY)
		rootNode->box.startY = sy;
	else
		sy = rootNode->box.startY;*/
	
	rootNode->box = boundingBox;
	rootNode->level=0;
	for(char i=0;i<MAX_CHILDREN;i++)
	{
		if(rootNode->myPoints.size() > 0)
		{
			donebuildsubtree = GAL_BuildSubTree(rootNode, i, 1, 1);
			if(donebuildsubtree != BUILDSUBTREE_SAMEPROCESS)
				flag  = false;
		}
	}
	if(flag)
		done=true;
	/*if(procRank == 0)
			printf("Num PLeaves %d NumExpectedUpdates %d\n",pLeafPointsPerProcess.size(), numExpectedUpdates);*/
	PBGL_oobMessage msg;
	int donetreeval;
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
			case MESSAGE_BUILDSUBTREE: 
				{
					MsgBuildSubTree msgBuildSubTree; 
				     	receive_oob(pg, msg.first, msg.second, msgBuildSubTree);
										
					if(msgBuildSubTree.hasMoreData)
					{
						receivedDataInBatches = true;
						msgSubtreeSecondLevel.ptv.insert(msgSubtreeSecondLevel.ptv.end(),msgBuildSubTree.ptv.begin(),msgBuildSubTree.ptv.end());
						break;
					}
					else
					{
						if(receivedDataInBatches)
							msgSubtreeSecondLevel.ptv.insert(msgSubtreeSecondLevel.ptv.end(),msgBuildSubTree.ptv.begin(),msgBuildSubTree.ptv.end());
					}
					
					GAL_Vertex *pRoot = GAL_CreateVertex(VTYPE_PSEUDOROOT);
					pRoot->parent = reinterpret_cast<GAL_Vertex*>(msgBuildSubTree.pLeaf[msgBuildSubTree.pLeaf.size()-1].second);
					pRoot->box = msgBuildSubTree.box;
					pRoot->parentDesc = msg.first;
					pRoot->level = msgBuildSubTree.depth;
					pRoot->pseudoRoot = true;
					((GAL_PseudoRoot*)pRoot)->childNo = msgBuildSubTree.childNo;

					TPointVector& tmpVector = receivedDataInBatches?msgSubtreeSecondLevel.ptv:msgBuildSubTree.ptv;
					TPointVector::iterator  iter = tmpVector.begin();
					for(;iter!=tmpVector.end();iter++)
					{
						pRoot->myPoints.push_back(*iter);
						pRoot->mass += iter->mass;
					}
					for(int i=0;i<MAX_CHILDREN;i++)
					{
						//DOR argument is subtreeHeight to restrict the quad-tree structure to two subtree deep.
						donebuildsubtree = GAL_BuildSubTree(pRoot, i, pRoot->level+1, subtreeHeight+1);
						assert(donebuildsubtree == BUILDSUBTREE_SAMEPROCESS);
					}
#ifdef SPAD_2
					((GAL_PseudoRoot*)pRoot)->pLeafLabel = msgBuildSubTree.pLeafLabel;
#endif
					/*if(msgBuildSubTree.depth == subtreeHeight)
					{
						assert(tmpNode->pseudoRoot);
						std::vector<std::pair<int, long int> >::iterator pLeafIter = msgBuildSubTree.pLeaf.begin();
						for(;pLeafIter!=msgBuildSubTree.pLeaf.end();pLeafIter++)
							((GAL_PseudoRoot*)tmpNode)->parents2.insert(*pLeafIter);
					}*/

					//If all the vertices of the subtree are owned by a single process, inform the caller process that subtree construction is done.
					if(donebuildsubtree == BUILDSUBTREE_SAMEPROCESS)
					{
						MsgUpdateMinMax msgDoneTree;
						//construct donesubtree message to notify parent
						msgDoneTree.pLeaf = reinterpret_cast<long int>(pRoot->parent);
						msgDoneTree.childNo = msgBuildSubTree.childNo;
						msgDoneTree.pRoot = reinterpret_cast<long int>(pRoot);
						//printf("Entire subtree belongs to single process.\n");
						GAL_SendMessage(msg.first,MESSAGE_DONESUBTREE, &msgDoneTree);
					}
					msgSubtreeSecondLevel.ptv.clear();
					receivedDataInBatches = false;
				}
				break;
			case MESSAGE_DONESUBTREE:
				{
				MsgUpdateMinMax msgDoneTree;
		              	receive_oob(pg, msg.first, msg.second, msgDoneTree);
				//Receive the node which is pseudo leaf.  A subtree of this node is fully constructed.
				GAL_Vertex* vert = reinterpret_cast<GAL_Vertex*>(msgDoneTree.pLeaf);
				GAL_Vertex* tmpVert = vert;
				
				//update the update count.
				numExpectedUpdates--;
					
				vert->pChild[msgDoneTree.childNo] = reinterpret_cast<GAL_Vertex*>(msgDoneTree.pRoot);	
				vert->childDesc[msgDoneTree.childNo] = msg.first;
				
				if(vert->level == (subtreeHeight-1))
				{
					MsgUpdatePLeaf msgUpdatePLeaf;
					msgUpdatePLeaf.label = vert->label;
					for(int i=0;i<MAX_CHILDREN;i++)
					{
						if(tmpVert->pChild[i])
						{
							if((tmpVert->childDesc[i]) == procRank)
								msgUpdatePLeaf.pChild[i] = 0;
							else
							{
								msgUpdatePLeaf.childDesc[i] = tmpVert->childDesc[i];
								msgUpdatePLeaf.pChild[i] = reinterpret_cast<long int>(tmpVert->pChild[i]);
							}
						}
					}
					vPLeaves.push_back(msgUpdatePLeaf);						
				}
				
				if(numExpectedUpdates == 0)
					done = true;

				if(done)
				{
					if(vPLeaves.size() > 4096)
					{
						printf("%d NUM_PSEUDOLEAVES %d \n",procRank,vPLeaves.size());
						GALHelper_WriteVertexToFile(vPLeaves);
					}
					else
						msgUpdatePLeaves.vPLeaves.insert(msgUpdatePLeaves.vPLeaves.begin(),vPLeaves.begin(), vPLeaves.end());
					for(int i=1;i<numProcs;i++)
					{
						GAL_SendMessage(i,MESSAGE_UPDATE_PLEAVES,&msgUpdatePLeaves); 
					}	
					vPLeaves.erase(vPLeaves.begin(),vPLeaves.end());
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
			case MESSAGE_UPDATE_PLEAVES:
				{
					MsgUpdatePLeaves msgUpdatePLeaves;
					receive_oob(pg, msg.first, msg.second, msgUpdatePLeaves);
					if(msgUpdatePLeaves.vPLeaves.size() == 0)
					{
						GALHelper_CreateVerticesFromFile(msgUpdatePLeaves);
						if(procRank==1)
							printf("%d NUM_PSEUDOLEAVES %d \n",procRank,msgUpdatePLeaves.vPLeaves.size());
					}
					int st = GAL_Aux_UpdatePLeaves(msgUpdatePLeaves);
					if(st == STATUS_SUCCESS)
						done = true;
					vPLeaves.erase(vPLeaves.begin(),vPLeaves.end());
				}
				break;
			default:break;
		}
		
	}

	if (numProcs==1)
		subtreeHeight=-1;
	synchronize(pg);
	/*if(procRank ==0)
		remove("ReplicatedTree.txt");*/
#ifdef SPAD_2
	GALHelper_ReplicateSubtrees();
#endif
	if(!hasMoreData)
	{
		pLeafMap.erase(pLeafMap.begin(),pLeafMap.end());
		msgUpdatePLeaves.vPLeaves.erase(msgUpdatePLeaves.vPLeaves.begin(),msgUpdatePLeaves.vPLeaves.end());
		pLeafPointsPerProcess.erase(pLeafPointsPerProcess.begin(),pLeafPointsPerProcess.end());
	
#ifdef TRAVERSAL_PROFILE
		local_subgraph<BGLGraph> local_g2(*g);
		vertices(local_g2);
		if (num_vertices(local_g2) > 0) 
		{
			printf("Num of vertices in process : %d = %ld\n",procRank,num_vertices(local_g2)); 
		}
		long int totalPRootLeaves = 0;
		reduce(communicator(pg),numPRootLeaves, totalPRootLeaves, std::plus<long int>(),0);
		if(procRank == 0)
		{
			printf("Total PRootLeaves:%ld\n",totalPRootLeaves);
		}
		
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
	}


	//printf("LabelCount:%d\n",labelCount);
	/*FILE* fp = fopen("treelog2.txt","w+");
	print_treetofile(fp);
	fclose(fp);*/
#ifdef PAPI
	int retval, EventSet=PAPI_NULL;
	long long values[4] = {(long long)0, (long long)0, (long long)0, (long long)0};

	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if(retval != PAPI_VER_CURRENT)
		handle_error(retval);
	
	retval = PAPI_multiplex_init();
	if (retval != PAPI_OK) 
		handle_error(retval);
	retval = PAPI_create_eventset(&EventSet);
	if (retval != PAPI_OK) 
		handle_error(retval);
 
	// Add Total L3Cache Misses 
	retval = PAPI_add_event(EventSet, PAPI_L3_TCM);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// Total L3 cache accesses = total memory accesses. Needed for computing L3 miss rate. On Wabash, there are 3 layers of cache. 
	retval = PAPI_add_event(EventSet, PAPI_L3_TCA);
	if (retval != PAPI_OK) 
		handle_error(retval);

	retval = PAPI_set_multiplex(EventSet);
	if (retval != PAPI_OK) 
		handle_error(retval);

	// TOTAL cycles 
	retval = PAPI_add_event(EventSet, PAPI_TOT_CYC);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// TOTAL instructions 
	retval = PAPI_add_event(EventSet, PAPI_TOT_INS);
	if (retval != PAPI_OK) 
		handle_error(retval);

#endif

	int globalDepth = 0;
	reduce(communicator(pg),depth, globalDepth, boost::parallel::maximum<int>(),0);
	long int numLocalPoints=0;
	std::vector<GAL_Vertex*>::iterator lIter = allLeaves.begin();
	for(;lIter!=allLeaves.end();lIter++)
		numLocalPoints += (*lIter)->myPoints.size();
	if(procRank == 0)
	{
		printf("%d Number of leaves %d points %ld\n",procRank,allLeaves.size(),numLocalPoints);
		printf("maximum depth : %d\n",globalDepth);
	}
#ifdef PAPI
	retval = PAPI_start(EventSet);
	if (retval != PAPI_OK) handle_error(retval);
#endif

	if(!hasMoreData)
	{
		double locs, loce;
		locs = clock();
		GALHelper_TraverseDown(rootNode, 2);
		//GALHelper_TraverseDown(rootNode, 3);
		loce = clock();
		double step2Time=(loce-locs)/CLOCKS_PER_SEC, maxStep2Time;
		locs = clock();
		GALHelper_TraverseUp(3);
		GALHelper_TraverseUp(4);
		loce = clock();
		double step34Time=(loce-locs)/CLOCKS_PER_SEC, maxStep34Time;
		reduce(communicator(pg),step2Time, maxStep2Time, boost::parallel::maximum<double>(),0);
		reduce(communicator(pg),step34Time, maxStep34Time, boost::parallel::maximum<double>(),0);
		
		if(procRank == 0)
		{
			printf("Top-down (step 2) traversal time %f\n",maxStep2Time);
			printf("Bottom up (steps 3 and 4 combined) traversal time %f\n",maxStep34Time);
		}

		/*printf("potential at all points\n",allLeaves.size());
		std::vector<GAL_Vertex*>::iterator iter = allLeaves.begin();
		for(;iter!=allLeaves.end();iter++)
		{
			std::vector<Point*>::iterator pIter = (*iter)->myPoints.begin();
			for(;pIter!=(*iter)->myPoints.end();pIter++)
			{
				printf("%f ",(*pIter)->potential);
			}
			printf("\n");
		}*/
	}
#ifdef PAPI
	/* Stop the counters */
	retval = PAPI_stop(EventSet, values);
	if (retval != PAPI_OK) 
		handle_error(retval);
#endif

#ifdef PAPI
	float missRate = values[0]/(double)(values[1]);
	float  CPI = values[2]/(double)(values[3]);
	printf("Average L3 Miss Rate:%f Average CPI (Total Cycles/ Total Instns): (%ld/%ld) = %f\n",missRate,values[2],values[3],CPI);
#endif

	return 0;
}

void GAL::GALHelper_DeleteSubtree(GAL_Vertex* node)
{
		assert(node != NULL);

		assert((node->desc).owner ==procRank); 
		
		if(node->isLeaf)
		{
#ifdef TRAVERSAL_PROFILE
			numLeavesAtHeight[node->level] -= 1;
			if(numLeavesAtHeight[node->level] < 0)
				numLeavesAtHeight[node->level] = 0;
#endif
			GAL_DeleteVertex(node);
			return;
		}
		
		for(int i=0;i<MAX_CHILDREN;i++)
		{
			if((node->childDesc[i] == procRank) && node->pChild[i])
				GALHelper_DeleteSubtree(node->pChild[i]);
			GAL_DeleteVertex(node);
		}
}

void GAL::GAL_DeleteVertex(GAL_Vertex* node)
{
	remove_vertex(node->desc, *g);
	delete node;
}

int GAL::GAL_BuildSubTree(GAL_Vertex* subtreeRoot, char index, int height, int DOR)
{
	if((subtreeRoot->isLeaf))
		return BUILDSUBTREE_SAMEPROCESS;
	else if((numProcs > 1) && (DOR == subtreeHeight))
	{
		bool messageSent = GALHelper_TestContainment(subtreeRoot, index);
		if(messageSent)
		{
			PLeafPointBucket bucket;
			bucket.pLeaf=reinterpret_cast<long int>(subtreeRoot);
			bucket.depth=height;
			pLeafPointsPerProcess.push_back(std::make_pair(subtreeRoot->label, bucket));
			return BUILDSUBTREE_FAILURE;
		}
		else
			return BUILDSUBTREE_SAMEPROCESS;

		
	}
	else
	{
		bool nodeCreated = false;
		//reached int node 
		int status = BUILDSUBTREE_SAMEPROCESS;
		//create a vertex for the intermediate node and add it to the tree
		GAL_Vertex* intNode = subtreeRoot->pChild[index]; 
		if(!intNode)
		{
			nodeCreated = true;
			if((DOR == 0)||(DOR==subtreeHeight))
				intNode = GAL_CreateVertex(VTYPE_PSEUDOROOT);
			else	
				intNode = GAL_CreateVertex(VTYPE_NORMAL);

			subtreeRoot->pChild[index] = intNode;
			subtreeRoot->childDesc[index] = procRank;
			intNode->level = height;
			intNode->parentDesc = (subtreeRoot->desc).owner;
			//Create edge between intermediate node and its parent and set properties
			//set parent pointer
			if((subtreeRoot->desc).owner == (intNode->desc).owner)
			{
				intNode->parent = subtreeRoot;
			}
			else
			{
				assert(0);
				intNode->parent = subtreeRoot->parent;
				intNode->pseudoRoot = true;
			}
		}

		float newnode_width = (subtreeRoot->box.endX - subtreeRoot->box.startX) /(float) 2;
		float newnode_height = (subtreeRoot->box.endY - subtreeRoot->box.startY) /(float) 2;
		newnode_width = ceil(newnode_width);
		newnode_height = ceil(newnode_height);
		assert(newnode_width == newnode_height);
		int sx, sy, ex, ey;
	
		switch(index)
		{
			case 1:
			      {
				sx = subtreeRoot->box.startX;
				sy = subtreeRoot->box.startY;
			      }
			      break;
			case 0:
			      {
				sx = subtreeRoot->box.startX + newnode_width;
				sy = subtreeRoot->box.startY;
			      }
			      break;
			case 2:
			      {
				sx = subtreeRoot->box.startX;
				sy = subtreeRoot->box.startY + newnode_height ;
			      }
			      break;
			case 3:
			      {
				sx = subtreeRoot->box.startX + newnode_width ;
				sy = subtreeRoot->box.startY + newnode_height ;
			      }
			      break;
		}
		ex = sx +  newnode_width ;
		ey = sy + newnode_height ;
		
		intNode->box.startX = sx;
		intNode->box.startY = sy;
		intNode->box.endX = ex;
		intNode->box.endY = ey;

		std::vector<Point>::iterator iter = subtreeRoot->myPoints.begin();
		for (int i=0;iter!=subtreeRoot->myPoints.end();i++) 
		{
			float x=(iter)->coordX;
			float y= (iter)->coordY;
			if((x >= sx)  && (x <= ex ) && (y >= sy)  && (y <= ey))
			//if (((iter)->coordX >= intNode->box.startX)  && ((iter)->coordX <= intNode->box.endX) && ((iter)->coordY >= intNode->box.startY)  && ((iter)->coordY <= intNode->box.endY))
			{
				intNode->mass += (iter)->mass;
				intNode->myPoints.push_back(*iter);
				iter = subtreeRoot->myPoints.erase(iter); 
			}
			else
				++iter;
		}
		if (((intNode->myPoints.size() <= NUM_POINTS_PER_CELL) ||(DOR == (MAX_LEVELS-1))))
		{
			if(nodeCreated)
			{
				intNode->isLeaf = true;
				if(intNode->myPoints.size() == 0)
				{
					GAL_DeleteVertex(intNode);
					subtreeRoot->pChild[index] = NULL;
					subtreeRoot->childDesc[index] = -1;
				}
				else
				{
					if(nodeCreated)
					{
						allLeaves.push_back(intNode);
						subtreeRoot->numChildren++;
						if(depth < intNode->level)
							depth  = intNode->level;
					}
				}
				return status;
			}
			else
			{
				if(intNode->isLeaf)
					return status;
			}
		}
		else if(intNode->isLeaf)
		{
			intNode->isLeaf = false;
			bool flag = false;
			std::vector<GAL_Vertex*>::iterator lIter = allLeaves.begin();
			for(;lIter!=allLeaves.end();lIter++)
			{
				if((*lIter) == intNode)
				{
					flag = true;
					allLeaves.erase(lIter);
					break;
				}
			}
			assert(flag == true);
		}
	
		if(nodeCreated)
			subtreeRoot->numChildren++;

		bool flag = false;
		for(char i=0;i<MAX_CHILDREN;i++)
		{
			//build left tree
			if(intNode->myPoints.size() > 0)
			{
				status = GAL_BuildSubTree(intNode,i,height+1, DOR+1);
				if(status == BUILDSUBTREE_FAILURE)
					flag = true;
				//We need to update the parent of the subtree rooted at intNode that left subtree construction is done or not done.
			}
		}
		//assert(intNode->myPoints.size() == 0);
		if(intNode->myPoints.size() != 0)
			printf("debug break\n");
		
		if(!flag)
		{
			status = BUILDSUBTREE_SAMEPROCESS;
			if(subtreeRoot && (subtreeRoot->desc).owner != (intNode->desc).owner)
			{
				MsgUpdateMinMax msgDoneTree;
				msgDoneTree.pRoot = reinterpret_cast<long int>(intNode);
				msgDoneTree.pLeaf = reinterpret_cast<long int>(subtreeRoot->parent);
				msgDoneTree.childNo = index;
				GAL_SendMessage((subtreeRoot->desc).owner,MESSAGE_DONESUBTREE,&msgDoneTree);
				status = BUILDSUBTREE_FAILURE;
			}
		}
		else
			status = BUILDSUBTREE_FAILURE;

		assert(status == BUILDSUBTREE_SAMEPROCESS);
		return status;
	}
}

GAL_Edge GAL::GAL_CreateEdge(GAL_Vertex& source, GAL_Vertex& target)
{
	GAL_Edge edge;
	std::pair<BGLEdgedesc,bool> p = add_edge(source.desc, target.desc,*g);
	
	edge.desc = p.first;
	return edge;
}

void GAL::GAL_SetEdgeProperties(GAL_Edge& edge, void* refedge, int vertexType)
{
	switch(vertextype)
	{
		case KDTREE: (*g)[edge.desc].left = reinterpret_cast<BGLTreeEdge*>(refedge)->left;
			break;
		default:break;
	}
}

void* GAL::GAL_GetEdgeProperties(GAL_Edge& e, int vertextype)
{
	BGLTreeEdge *ret = NULL;
	switch(vertextype)
	{
		case KDTREE: 
			ret = new BGLTreeEdge();
			ret->left = (*g)[e.desc].left; 				
			break;
		default: break;

	}
	return ret;
}

GAL_Vertex* GAL::GAL_GetStartVertex()
{
	return rootNode;
}

void GAL::GAL_PrintGraph()
{
#ifdef TRAVERSAL_PROFILE
	long int totalPointsVisited =0;
	//to find the bottlenecks uncomment the below line and all its occurences.
	//std::vector<std::pair<long int, long int> > footprintMatrix;
	std::vector<long int>::iterator iter = pipelineStage.begin();
	for(;iter!=pipelineStage.end();iter++)
	{
		GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(*iter);
		if((pRoot->level != 0))// && (pRoot->leftChild || pRoot->rightChild))
		{
			long int count=0;
			GALHelper_CountSubtreeNodes(pRoot, count);
			//printf("%d Subtree %ld Points_Visited %ld\n",procRank, pRoot->label,count);
			//footprintMatrix.push_back(std::make_pair(*iter,count));
			totalPointsVisited += count;
		}
		/*else if(pRoot->level == 0)
		{
			long int count=0;
			GALHelper_CountSubtreeNodes(pRoot, count, true);
			printf("%d Subtree %ld Points_Visited %ld\n",procRank, *iter,count);
			totalPointsVisited += count;
		}*/
	}
	long int count=0;
	GALHelper_CountSubtreeNodes(rootNode, count);
	printf("Subtree %ld Points_Visited %ld\n",reinterpret_cast<long int>(rootNode),count);
	double bof=0.;
	GALHelper_GetBlockOccupancyFactor(rootNode, bof);
	printf("Bof:  %f\n",bof/count);

	//printf("%d: number of subtrees:%d average number of visitors:%f\n",procRank,pipelineStage.size(),totalPointsVisited/(float)pipelineStage.size());
	//Finding bottlenecks.
	/*std::sort(footprintMatrix.begin(),footprintMatrix.end(),SortLongIntPairsBySecond_Decrease);
	int numberOfBottlenecks=1008,j=0;
	while(j < numberOfBottlenecks)
	{
		int i=0;
		long int localElem=-1, globalElem;
		if(i<footprintMatrix.size())
			localElem = footprintMatrix[i].second;	
		all_reduce(communicator(pg),localElem,globalElem,boost::parallel::maximum<long int>());
		if(i<footprintMatrix.size())
		{
			if(localElem == globalElem)
			{
				GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(footprintMatrix[i].first);
				int childNum = pRoot->isLeftChild?0:1;
				footprintMatrix.erase(footprintMatrix.begin());
				printf("%d %ld %d\n",procRank,((GAL_PseudoRoot*)pRoot)->pLeafLabel, childNum);
			}	
		}	
		j++;
	}*/
#endif

#ifdef STATISTICS2
	/*local_subgraph<BGLGraph> local_g2(*g);
      vertices(local_g2);
      if (num_vertices(local_g2) > 0) {
	printf("Num of vertices in process : %d = %ld\n",procRank,num_vertices(local_g2)); 
	}*/
	int totalBlocks=0;
	reduce(communicator(pg),numBlocks, totalBlocks, std::plus<int>(),0);
	double totalParOverhead=0., avgParOverhead=parOverhead;///(double)numBlocks;
	//reduce(communicator(pg),avgParOverhead, totalParOverhead, std::plus<double>(),0);
	reduce(communicator(pg),avgParOverhead, totalParOverhead, boost::parallel::maximum<double>(),0);
	if(procRank == 0)
		//printf("Parallel Overhead (per block - across all processes):%f\n",(totalParOverhead/numProcs)/CLOCKS_PER_SEC); 
		printf("Parallel Overhead (max for any block - across all processes):%f\n",totalParOverhead/CLOCKS_PER_SEC); 
	//printf("%d: Parallel Overhead (per block):%f\n",procRank,avgParOverhead/CLOCKS_PER_SEC); 
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
		
	/*std::vector<std::pair<int, int> > tmpTruncatedList;
	std::map<int, std::vector<int> >::iterator tmpMIter = nodeSeenAtChkPt.begin();
	for(;tmpMIter!=nodeSeenAtChkPt.end();tmpMIter++)
	{
		if(tmpMIter->second.size() !=0)
		{
			std::vector<int>::iterator tmpVIter = (tmpMIter->second).begin();
			//tmpTruncatedList.push_back(std::make_pair(tmpMIter->first,(tmpMIter->second)[tmpMIter->second.size()-1]));
			for(;tmpVIter!=(tmpMIter->second).end();tmpVIter++)
			{
				//printf("%d, ",*tmpVIter);
				tmpTruncatedList.push_back(std::make_pair(tmpMIter->first,*tmpVIter));
			}
			//printf("\n");
		}
		
	}
	std::sort(tmpTruncatedList.begin(), tmpTruncatedList.end(), SortIntPairsBySecond);
	std::vector<std::pair<int, int> >::iterator truIter=tmpTruncatedList.begin();
	for(;truIter!=tmpTruncatedList.end();truIter++)
	{
		//printf("%d %d\n",truIter->second,truIter->first);
	}*/

	//printf("%d: messages processed:%ld, local PPM:%f, workDone:%ld \n",procRank,traverseCount,localPPM,workDone);
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
#ifdef MESSAGE_AGGREGATION
		TBlockId fragBlkId;
		bool readyToSend=false;	

		if(tmpChildNode->pseudoLeaf) //TODO: && tmpChildNode->isLeftChild)
		{	
			(tmpChildNode->numBlocksVisited)--;
			if(tmpChildNode->numBlocksVisited < 0)
				tmpChildNode->numBlocksVisited=0;
		}

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
		int procId = (tmpChildNode->parentDesc);
		if(tmpChildNode->pseudoRoot && (tmpChildNode->level == subtreeHeight))
		{
			BlockStack* curBStack = reinterpret_cast<BlockStack*>(curBlockId.second);
			procId = curBStack->parentBlockId.first;
			assert(procId < numProcs);
			/*std::vector<RepPLeaf>::iterator pLeafIter=(((GAL_PseudoRoot*)(tmpChildNode))->parents).begin();
			for(;pLeafIter!= (((GAL_PseudoRoot*)(tmpChildNode))->parents).end();pLeafIter++)
			{
				if((pLeafIter->first) == procId)
				{
					parent = pLeafIter->second;
					break;
				}
			}*/
			parent = ((GAL_PseudoRoot*)(tmpChildNode))->parents2[procId];
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
#ifndef HYBRID_BUILD
					if((nextNodeToVisit->pseudoRoot)&& (nextNodeToVisit->level == subtreeHeight))
						((GAL_PseudoRoot*)nextNodeToVisit)->parents2[msgTraverse.blkStart.first] = msgTraverse.pLeaf;
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
				int nextNodeProc = -1; //TODO
				/*if(tmpChildNode->isLeftChild)
				{
					nextNodeProc = prnt->rightDesc;
				}
				else
				{
					nextNodeProc = 	prnt->leftDesc;
				}*/
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
#ifndef HYBRID_BUILD
					if((nextNodeToVisit->pseudoRoot)&& (nextNodeToVisit->level == subtreeHeight))
						((GAL_PseudoRoot*)nextNodeToVisit)->parents2[msgTraverse.blkStart.first] = msgTraverse.pLeaf;
#endif
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
	
	int ret =STATUS_TRAVERSE_COMPLETE;
	int status = STATUS_SUCCESS;
#ifdef PERFCOUNTERS
	bool found=false;
	std::pair<std::map<long int, PipelineStats>::iterator,bool> doesBlockExist;
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
						for(int i=0;i<numProcs;i++)
						{
							if(i!=procRank)
								GAL_SendMessage(i,MESSAGE_SENDCOMPRESSED,&sentVal); 
						}
					}
#endif
				}
			}
			else
			{
				//ret = GAL_TraverseHelper(vis,rootNode, NULL);
				
				if(ret == STATUS_TRAVERSE_INCOMPLETE)
				{
					status = STATUS_FAILURE;
				}
				else if(ret == STATUS_TRAVERSE_COMPLETE)
				{
					int re = vis->GALVisitor_RemoveBlockStack();		
						if(re == -1)
							printf("Debug 5\n");
				}

			}
		}
		while(status == STATUS_SUCCESS);
	}


#ifdef PERFCOUNTERS
	if(compTimer.IsRunning())
		compTimer.Stop();
	if(found)
	{
		uint64_t stageExTime = compTimer.GetLastExecTime();
		((doesBlockExist.first)->second).timeInStage += stageExTime/(float)CLOCKS_PER_SEC;
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
	/*for(int i=0;i<25;i++)
	{
		std::vector<int> tmpVector;
		nodeSeenAtChkPt.insert(std::make_pair(i,tmpVector));
	}*/

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
		PBGL_AsyncMsg pollMsg = pg.poll();
		if(!pollMsg)
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
							for(int i=0;i<numProcs;i++)
							{
								if(i !=procRank)
									GAL_SendMessage(i,MESSAGE_DONEKDTREE,&msgTerminate); 
							}
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
						GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(msgTraverse.pRoot);
						GAL_PseudoRoot* pSibling = reinterpret_cast<GAL_PseudoRoot*>(msgTraverse.pSibling);
						bool loadBalanced = false;
						int procId = msgTraverse.blkStart.first;
						if(procId == procRank)
						{
							loadBalanced = true;
						}
#ifndef HYBRID_BUILD
						if(pRoot->pseudoRoot)
							((GAL_PseudoRoot*)pRoot)->parents2[procId]=msgTraverse.pLeaf;
#endif

#ifdef PERFCOUNTERS
						compTimer.Start();
						if(pRoot->level == (2* subtreeHeight))
							leafStageTimer.Start();
#endif
#ifdef STATISTICS
						traverseCount++;
						pointsProcessed += msgTraverse.l.size();
#endif
						/*if(pSibling)
						{
							((GAL_PseudoRoot *)(pRoot))->pSibling = pSibling;
							((GAL_PseudoRoot *)(pRoot))->siblingDesc = msgTraverse.siblingDesc;
						}*/

						/*if(msgTraverse.blkStart.second == 67110524)
						{
							printf("Debug 9\n");
						}*/
						
						
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
									long int parent = reinterpret_cast<long int>(pRoot->parent);
									/*std::vector<RepPLeaf>::iterator pLeafIter=(((GAL_PseudoRoot*)(pRoot))->parents).begin();
									for(;pLeafIter!= (((GAL_PseudoRoot*)(pRoot))->parents).end();pLeafIter++)
									{
										if(procRank == 0)
											break;
										if((pLeafIter->first) == procRank)
										{
											parent = pLeafIter->second;
											break;
										}
									}*/
									if((pRoot->pseudoRoot) && (procRank != 0))
										parent = ((GAL_PseudoRoot*)(pRoot))->parents2[procRank];

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
						((doesBlockExist.first)->second).timeInStage += stageExTime/(float)CLOCKS_PER_SEC;
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
						std::pair<std::map<long int, PipelineStats>::iterator,bool> doesBlockExist;
						compTimer.Start();
#endif
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
									if(re == -1)
											printf("Debug 8\n");
							}
						}
#ifdef PERFCOUNTERS
						if(compTimer.IsRunning())
							compTimer.Stop();
						uint64_t stageExTime = compTimer.GetLastExecTime();
						((doesBlockExist.first)->second).timeInStage += stageExTime/(float)CLOCKS_PER_SEC;
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
				case MESSAGE_DONEKDTREE:
				{
					MsgTerminate msgTerminate;
						receive_oob(pg, msg.first, msg.second, msgTerminate);
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
	leafStageTime = leafStageTimer.GetTotTime();
#endif	
	return ret;
}

int GAL::GAL_TraverseHelper(GALVisitor* vis, GAL_Vertex* node, GAL_Vertex* sib)
{
#if 0
		TIndices leftBlock, rightBlock, superBlock;
		int ret = STATUS_TRAVERSE_COMPLETE;
		TBlockId blockId;
		long int uniqBlkId1, uniqBlkId2;
		if(node==0)
			assert(false);//return ret;
		assert((node->desc).owner ==procRank); 
		BlockStackListIter rightBStack, curBStack = vis->GALVisitor_GetCurrentBlockStack();
		
		bool traverseSubTree = vis->GALVisitor_VisitNode(node, sib, leftBlock, rightBlock, blockId);
		if(!traverseSubTree)
		{
			/*if((node->leftChild !=NULL) || (node->rightChild!=NULL))
			{
				(nodeSeenAtChkPt[node->level]).push_back(vis->GALVisitor_GetNodeCountOfSinglePoint());
			}*/
			//printf("%d\n",node->level);
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
				if(node->leftChild)
				{
#ifndef HYBRID_BUILD
					if((node->leftChild->level==subtreeHeight) && (node->leftChild->leftChild || node->leftChild->rightChild))
					{
						GAL_PseudoRoot* pRoot = (GAL_PseudoRoot*)(node->leftChild);
						pRoot->parents2[procRank] = reinterpret_cast<long int>(node);
					}
#endif
						ret = GAL_TraverseHelper(vis,node->leftChild, node->rightChild);
				}
				if(ret == STATUS_TRAVERSE_COMPLETE)
				{
					if(node->rightChild && ((node->rightDesc) != procRank))
					{	
#ifdef PERFCOUNTERS
						compTimer.Stop();
#endif
						//assert(0);
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
						if(node->rightChild)
						{
#ifndef HYBRID_BUILD
							if((node->rightChild->level==subtreeHeight) && (node->rightChild->leftChild || node->rightChild->rightChild))
							{
								GAL_PseudoRoot* pRoot = (GAL_PseudoRoot*)(node->rightChild);
								pRoot->parents2[procRank] = reinterpret_cast<long int>(node);
							}
#endif
							ret = GAL_TraverseHelper(vis,node->rightChild, NULL);
						}
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
#endif
}
#ifdef MESSAGE_AGGREGATION
/* Description: This function is called at a pseudo leaf to compress messages.
 * Prerequisites: The blockID is already removed from traversalready queue.
 * Parameters: msgTraverse - INOUT - reference to a MsgTraverse. It contains the aggregated vector if the buffer length exceeds PIPELINE_BUFFER_SIZE
 * 		blkSTart - IN - block ID of the block relative to which the pointers present in msgTraverse.l are valid.
 * 		node - IN - pseudo leaf node whose left child needs to be visited.
 * Return Value: size of the aggregated buffer at this node.
 */
int GAL::GAL_TraverseHelper_CompressMessages(GALVisitor* vis,MsgTraverse& msgTraverse, int aggrBlockSize, MsgTraverse& msgTraverseRight, bool goingLeft)
{
#if 0
	TBlockId blockId;
	TIndices tmpIndices;
	long int pseudoLeaf;
	if(goingLeft)
		pseudoLeaf = msgTraverse.pLeaf;
	else
		pseudoLeaf = msgTraverseRight.pLeaf;

	GAL_Vertex* pLeaf = reinterpret_cast<GAL_Vertex*>(pseudoLeaf);	
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

	//if((aggrSize >= aggrBlockSize) || (pLeaf->numBlocksVisited <=0))
	if(aggrSize >= aggrBlockSize)
	{
		vis->GALVisitor_SetAsCurrentBlockStack2(bufferStack);
		msgTraverse.l.clear();
		msgTraverseRight.l.clear();
		blockId = vis->GALVisitor_GetBufferData(msgTraverse.l, msgTraverseRight.l);
		msgTraverse.blkStart = blockId;
		msgTraverseRight.blkStart = blockId;
		aggrBuffer.erase(bufIter);
		/*if(pLeaf->numBlocksVisited <=0)
			aggrSize=aggrBlockSize;*/
	}

	return aggrSize;
#endif
}


/* Description: This function is called to send compressed messagess.
 * compressed message buffers may exist at multiple pseudo leaves (At different heights) within the same process. Some may have been processed and some may not.
 * Hence it is necessary to go through entire aggrBuffer to and then send all the buffers that have not been processed.
 * Parameters: None
 * Return Value: None
 */
void GAL::GAL_TraverseHelper_SendCompressedMessages(GALVisitor* vis)
{
#if 0
	//search for the entry in the buffer for this 'node'
	std::map<long int, long int>::iterator bufIter = aggrBuffer.begin();
	while(bufIter != aggrBuffer.end())
	{
		MsgTraverse msgTraverse, msgTraverseRight;
		TBlockId blockId;
		GAL_Vertex* pLeaf = reinterpret_cast<GAL_Vertex*>(bufIter->first);
		/*if(!(pLeaf->isLeftChild) && (pLeaf->numBlocksVisited>0))
		{
			bufIter++;
			continue;
		}*/
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
#endif
		aggrBuffer.erase(bufIter);
		bufIter++;
	}

	return;	
#endif
}

/* Description: This function is called to handle MESSAGE_TRAVERSE_BACKWARD message containing encoding of multiple blocks. 
 * 		The object containing pseudoLeaf is removed from the list as well.
 * Parameters:  vis = reference to visitor object.
 *             blkIdSet - OUT- set containing IDs of the block whose elements were aggregated and composed as a single block. 
 * Return Value: status of traversal
 */

int GAL::GALHelper_HandleMessageTraverseBackward_Multiple(GAL_Vertex* parentNode, GALVisitor* vis, TCBSet& blkIdSet)
{
#if 0
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
	#endif
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
					if(tmpVertex->level == (2*subtreeHeight))
						leafStageTimer.Stop();
					compTimer.Stop();
#endif
					//sending entire compressed data to sibling
					msgTraverse.pLeaf = msgTraverse.pLeaf;
					msgTraverse.pRoot = msgTraverse.pSibling;
					msgTraverse.pSibling = static_cast<long int>(0);
					if(!loadBalanced)
					{
						msgTraverse.blkStart = blkStart;
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
#ifndef HYBRID_BUILD
					if((*pRoot)->level==subtreeHeight)
						((GAL_PseudoRoot*)(*pRoot))->parents2[msgTraverse.blkStart.first] = msgTraverse.pLeaf;
#endif
					vis->GALVisitor_AddToSuperBlock(*pRoot, msgTraverse.l, msgTraverse.blkStart, NULL, 0);
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

int GAL::GAL_Aux_UpdatePLeaves(MsgUpdatePLeaves& msg)
{
	int ret = STATUS_SUCCESS;
	std::vector<MsgUpdatePLeaf>::iterator msgIter = msg.vPLeaves.begin();

	for(msgIter = msg.vPLeaves.begin();msgIter != msg.vPLeaves.end();msgIter++)
	{
		GAL_Vertex* pLeaf = NULL;
		std::vector<std::pair<int, PLeafPointBucket> >::iterator pLeafIter;
		for(pLeafIter = pLeafPointsPerProcess.begin();pLeafIter!=pLeafPointsPerProcess.end();pLeafIter++)
		{
			if(pLeafIter->first == msgIter->label)
			{
				pLeaf = reinterpret_cast<GAL_Vertex*>((pLeafIter->second).pLeaf);
				break;
			}
		}	
		assert(pLeaf);
		GAL_Vertex* tmpVert = pLeaf;
		for(int i=0;i<MAX_CHILDREN;i++)
		{
			if((pLeaf->pChild[i] == NULL) && msgIter->pChild[i])
			{
				pLeaf->childDesc[i] = msgIter->childDesc[i];	
				pLeaf->pChild[i] = reinterpret_cast<GAL_Vertex*>(msgIter->pChild[i]);
				if(pLeaf->childDesc[i] == procRank)
				{
					GAL_PseudoRoot* child = reinterpret_cast<GAL_PseudoRoot *>(msgIter->pChild[i]);
					child->parents2[procRank]=(pLeafIter->second).pLeaf;
				}
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
#ifdef SPAD_2
	numAdjacentSubtrees++;
#endif
	std::pair<std::map<long int, int>::iterator, bool> ret = pLeafMap.insert(std::make_pair(pLeaf,0));
	if(ret.second == true)
		(ret.first)->second = nextprocessid;

	/*int nextprocessid;
	std::pair<std::map<long int, int>::iterator, bool> ret = pLeafMap.insert(std::make_pair(pLeaf,0));
	if(ret.second == false)
		return (ret.first)->second;
	else
	{
		do
		{
				nextprocessid = ((procRank)* 2 + 1 + procCount) % numProcs;
				procCount++;
				if(procCount > (1<<(subtreeHeight)))
					procCount = 1;
		}while(nextprocessid == procRank);

	}
	(ret.first)->second = nextprocessid;*/
	return nextprocessid;
}

void GAL::GALHelper_CountSubtreeNodes(GAL_Vertex* ver, long int& count, bool isRootSubtree)
{
		int ret = STATUS_TRAVERSE_COMPLETE;
#ifdef TRAVERSAL_PROFILE
		count += 1;//ver->pointsVisited;
#else
		count++;
#endif
		assert(ver != NULL);

		assert((ver->desc).owner ==procRank); 
		
		//printf("%d %d %f %f %f %f\n",ver->label, ver->level, ver->box.startX, ver->box.startY, ver->box.endX, ver->box.endY);

		for(int i=0;i<MAX_CHILDREN;i++)
		{
			if(ver->pChild[i])
			{
				if(ver->childDesc[i] == procRank)
				{
					GAL_Vertex* tmp=ver->pChild[i];
					if((numProcs==1) || !isRootSubtree)
						GALHelper_CountSubtreeNodes(tmp,count, isRootSubtree);
					else if((tmp->level < subtreeHeight) || tmp->isLeaf)
						GALHelper_CountSubtreeNodes(tmp,count, isRootSubtree);
				}
			}
		}
			
		
}

void GAL::print_treetofile(FILE* fp)
{
	print_preorder(rootNode, fp);
}

void GAL::print_preorder(GAL_Vertex* node, FILE* fp)
{
	//fprintf(fp,"%d %d %f %f %f %f\n",node->label, node->level, node->box.startX, node->box.startY, node->box.endX, node->box.endY);
	if(node->isLeaf)
		fprintf(fp,"%d %d %d\n",node->label, node->level, node->myPoints.size());
	for(int i=0;i<MAX_CHILDREN;i++)
		if(node->pChild[i])
			print_preorder(node->pChild[i],fp);
}

void GAL::GALHelper_WriteVertexToFile(std::vector<MsgUpdatePLeaf>& msg)
{
	std::ofstream output;
	output.open("ReplicatedTree.txt", std::fstream::out);
	//FILE* fp = fopen("ReplicatedTree.txt","w");
	//std::vector<MsgUpdatePLeaf>::iterator msgIter = msg.vPLeaves.begin();
	std::vector<MsgUpdatePLeaf>::iterator msgIter = msg.begin();
	
	//for(msgIter = msg.vPLeaves.begin();msgIter != msg.vPLeaves.end();msgIter++)
	for(msgIter = msg.begin();msgIter != msg.end();msgIter++)
	{
		output<<msgIter->label<<" ";
		for(int i=0;i<MAX_CHILDREN;i++)
			output<<msgIter->pChild[i]<<" "<<(msgIter->childDesc[i])<<" ";
		//fprintf(fp,"%ld %ld %d %ld %d",msgIter->label,msgIter->pLeftChild,msgIter->leftDesc,msgIter->pRightChild,msgIter->rightDesc);
		//fprintf(fp,"\n");
		output<<std::endl;
	}
	//fclose(fp);
	output.close();
}

void GAL::GALHelper_CreateVerticesFromFile(MsgUpdatePLeaves& msgUpdatePLeaves)
{
	/*FILE* fp=fopen("ReplicatedTree.txt","r");
	while (!feof(fp)) 
	{ 
		MsgUpdatePLeaf p;
		int status = fscanf(fp, "%ld %ld %d %ld %d", p.label, p.pLeftChild,p.leftDesc, p.pRightChild,p.rightDesc);
		if(status == 2)
			break;
		for(int i=0;i<DIMENSION;i++)
		{
			status = fscanf(fp, "%f %f", p.maxd[i],p.mind[i]);
			if(status == 2)
				break;
		}
		if(status == 2)
			break;
		msgUpdatePLeaves.vPLeaves.push_back(p);
	}
	fclose(fp);*/
	std::ifstream input("ReplicatedTree.txt", std::fstream::in);
	if(input.fail())
	{
		std::cout<<"ERROR! Subtree Replication Failed exiting"<<std::endl;
		MPI_Finalize();
		exit(0);
	}
	if ( input.peek() != EOF ) 
	{
	        while(true) 
		{
			MsgUpdatePLeaf pLeaf;
			int label,childDesc;
			long int pChild;
			input >> label;
			for(int i=0;i<MAX_CHILDREN;i++)
			{
				input >> pChild;
				input >> childDesc;
				pLeaf.pChild[i] = pChild;
				pLeaf.childDesc[i] = childDesc;
			}
			pLeaf.label=label;
			if(input.eof())
			{
				break;
			}
			msgUpdatePLeaves.vPLeaves.push_back(pLeaf);
		}
	}
	input.close();
}

void GAL::GALHelper_SaveSubtreeInFile(GAL_Vertex* node, long int parentId, int childNum, std::ofstream& fp)
{
	assert(node != NULL);

	assert((node->desc).owner ==procRank); 

	int pLeaf = 0;
	if(node->pseudoRoot)
		pLeaf=2; //special value to capture pseudoroot;
	int isLeaf=node->isLeaf?1:0;
	for(int i=0;i<MAX_CHILDREN;i++)
	{
		if(node->pChild[i] && (node->childDesc[i] != procRank))
			pLeaf = 1;
	}	
	fp<<node->label<<" "<<parentId<<" "<<childNum<<" "<<(short int)(node->level)<<" "<<pLeaf<<" "<<isLeaf<<" "<<node->box.startX<<" "<<node->box.startY<<" "<<node->box.endX<<" "<<node->box.endY;
	if(pLeaf==1)
	{
		for(int i=0;i<MAX_CHILDREN;i++)
			fp<<" "<<reinterpret_cast<long int>(node->pChild[i])<<" "<<node->childDesc[i];
	}
	if(isLeaf)
	{
		fp <<" "<<node->myPoints.size();
		for(int i=0;i<node->myPoints.size();i++)
		{
			fp<<" "<<(node->myPoints[i]).id<<" "<<(float)(node->myPoints[i].coordX)<<" "<<(float)(node->myPoints[i].coordY);
		}
	}
	fp<<std::endl;
	if(isLeaf)
	{
		return;
	}
	
	for(int i=0;i<MAX_CHILDREN;i++)
	{
		if(node->pChild[i] && (node->childDesc[i] == procRank))
		{
			GALHelper_SaveSubtreeInFile(node->pChild[i],node->label,0,fp);
		}
	}
	
	return;

}

void GAL::GALHelper_CreateSubtreeFromFile(GAL_Vertex** pRoot, GAL_Vertex* pLeaf, int pLeafOwner)
{
	std::map<long int,long int> repVertexTable;
	int recordNum = 0;

	std::ifstream input("ReplicatedTree.txt", std::fstream::in);
	if(input.fail())
	{
		std::cout<<"ERROR! Subtree Replication Failed exiting"<<std::endl;
		MPI_Finalize();
		exit(0);
	}
	if ( input.peek() != EOF ) 
	{
	        while(true) 
		{
			int childNum, isLeaf=0, isPLeaf=0;
			long int label, parentId;
			short int level;
			float sx, sy, ex, ey;

			recordNum++;
				
			GAL_Vertex *parentNode, *node=NULL;
			input >> label;
			if(input.eof())
			{
				break;
			}
			input >> parentId;
			input >> childNum;
			input >> level;
			input >> isPLeaf;
			input >> isLeaf;
			input >> sx;
			input >> sy;
			input >> ex;
			input >> ey;
			


			if(recordNum == 1)
			{
				parentNode=pLeaf;
				node = GAL_CreateVertex(VTYPE_PSEUDOROOT);
				node->pseudoRoot = true;
				*pRoot=node;
				node->parentDesc = pLeafOwner;
			}
			else
			{
				if((isPLeaf==2)||(isLeaf==1))
				{
					node = GAL_CreateVertex(VTYPE_PSEUDOROOT);
					node->pseudoRoot = true;
#ifdef TRAVERSAL_PROFILE
					pipelineStage.pop_back();
#endif
				}
				else
					node = GAL_CreateVertex(VTYPE_NORMAL);

				parentNode = reinterpret_cast<GAL_Vertex*>(repVertexTable[parentId]);
				node->parentDesc = procRank;
			}
			
			node->box.startX = sx;
			node->box.startY = sy;
			node->box.endX = ex;
			node->box.endY = ey;
			node->label=label;
			node->level=level;
			node->parent = parentNode;
			if(recordNum != 1)
			{
				parentNode->pChild[childNum]=node;
				parentNode->childDesc[childNum] = procRank;
			}

			long int vertexPtr = reinterpret_cast<long int>(node);
			repVertexTable.insert(std::make_pair(label,vertexPtr));	

			if(isLeaf)
			{
				node->isLeaf = true;
				int numPoints = 0;
				input >> numPoints;
				for(int i=0;i<numPoints;i++)
				{
					Point pt;
					input >> pt.id;
					input >> pt.coordX;
					input >> pt.coordY;
					node->myPoints.push_back(pt);	
				}
			}
			else
			{
				if(isPLeaf == 1)
				{
					for(int i=0;i<MAX_CHILDREN;i++)
					{
						long int pChild;
						int childDesc;
						input>>pChild;
						input>>childDesc;
						node->pChild[i] = reinterpret_cast<GAL_Vertex*>(pChild);
						node->childDesc[i] = childDesc;
					}
				}
			}

		}
	}
	input.close();
}

void GAL::GALHelper_SaveSubtreeAsString(GAL_Vertex* subtreeRoot, std::vector<std::string>& subtreeStr, long int parentId, int childNum, bool requestPending)
{
	assert(subtreeRoot != NULL);

	GAL_Vertex* node = subtreeRoot;
	assert((node->desc).owner ==procRank); 

	std::stringstream stroutput;
	int pLeaf = 0;
	if(node->pseudoRoot)
		pLeaf=2; //special value to capture pseudoroot;
	for(int i=0;i<MAX_CHILDREN;i++)
	{
		if(node->pChild[i] && (node->childDesc[i] != procRank))
			pLeaf = 1;
	}
	int isLeaf=(node->isLeaf)?1:0;
	assert(pLeaf != 1);
	
	stroutput<<node->label<<" "<<parentId<<" "<<childNum<<" "<<(short int)(node->level)<<" "<<pLeaf<<" "<<isLeaf<<" "<<node->box.startX<<" "<<node->box.startY<<" "<<node->box.endX<<" "<<node->box.endY;
	if(pLeaf==1)
	{
		for(int i=0;i<MAX_CHILDREN;i++)
			stroutput<<" "<<reinterpret_cast<long int>(node->pChild[i])<<" "<<node->childDesc[i];
	}
	if(isLeaf)
	{
		stroutput <<" "<<node->myPoints.size();
		for(int i=0;i<node->myPoints.size();i++)
		{
			stroutput<<" "<<(node->myPoints[i]).id<<" "<<(float)(node->myPoints[i].coordX)<<" "<<(float)(node->myPoints[i].coordY);
		}
	}
	subtreeStr.push_back(stroutput.str());

	if(isLeaf || requestPending)
	{
		return;
	}
	for(int i=0;i<MAX_CHILDREN;i++)
	{
		if(node->pChild[i] && (node->childDesc[i] == procRank))
		{
			GALHelper_SaveSubtreeAsString(node->pChild[i],subtreeStr,node->label,0,false);
		}
	}
	
	return;
}

void GAL::GALHelper_ReplicateSubtreeFromString(MsgReplicateSubtree& msgCloneSubtree, GAL_Vertex** pRoot, GAL_Vertex* pLeaf, int pLeafOwner)
{
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
		float sx, sy, ex, ey;

		recordNum++;
			
		GAL_Vertex *parentNode, *node=NULL;
		strinput >> label;
		strinput >> parentId;
		strinput >> childNum;
		strinput >> level;
		strinput >> isPLeaf;
		strinput >> isLeaf;
		strinput >> sx;
		strinput >> sy;
		strinput >> ex;
		strinput >> ey;


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
#ifdef TRAVERSAL_PROFILE
				pipelineStage.pop_back();
#endif
			}
			else
				node = GAL_CreateVertex(VTYPE_NORMAL);

			parentNode = reinterpret_cast<GAL_Vertex*>(repVertexTable[parentId]);
		}
		
		node->label=label;
		node->level=level;
		node->parentDesc = procRank;
		node->parent = parentNode;
		node->box.startX = sx;
		node->box.startY = sy;
		node->box.endX = ex;
		node->box.endY = ey;
		if(recordNum != 1)
		{
			parentNode->pChild[childNum]=node;
			parentNode->childDesc[childNum] = procRank;
		}

		long int vertexPtr = reinterpret_cast<long int>(node);
		repVertexTable.insert(std::make_pair(label,vertexPtr));	

		if(isPLeaf==1)
			assert(0);
		
		if(isLeaf)
		{
			node->isLeaf = true;
			int numPoints = 0;
			strinput >> numPoints;
			for(int i=0;i<numPoints;i++)
			{
				Point pt;
				strinput >> pt.id;
				strinput >> pt.coordX;
				strinput >> pt.coordY;
				node->myPoints.push_back(pt);	
			}
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
}



void GAL::GALHelper_ReplicateTopSubtree()
{
	long int count=0;
	MsgReplicateSubtree subtree;
	GALHelper_CountSubtreeNodes(rootNode, count);
	if(count > 4096)
	{
		printf("Creating top subtree in file\n");
		std::ofstream output;
		output.open("ReplicatedTree.txt", std::fstream::out);
		GALHelper_SaveSubtreeInFile(rootNode,0,0,output);
		output.close();
	}
	else
		GALHelper_SaveSubtreeAsString(rootNode, subtree.data, 0, 0, false);

	subtree.pLeaf=0;
	for(int i=1;i<numProcs;i++)
		GAL_SendMessage(i,MESSAGE_CREATE_SUBTREE, &subtree);
}


#ifdef SPAD_2
void GAL::GALHelper_ReplicateSubtrees()
{
	PBGL_oobMessage msg;
	int numLocalReplicas=0;
	bool done=false;
	if(numAdjacentSubtrees > 0)
		printf("Number of subtrees adjacent to process %d:%d\n", procRank,numAdjacentSubtrees);
	if(procRank==0)
	{
		int status=GALHelper_GetRandomSubtreeAndBroadcast();
		//int status=GALHelper_GetBottleneckSubtreesAndBroadcast(123,"bneck123.txt");
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
					if(pRoot == NULL)
					{
						GAL_Vertex* pLeaf = NULL;
						std::vector<std::pair<int, PLeafPointBucket> >::iterator pLeafIter;
						for(pLeafIter = pLeafPointsPerProcess.begin();pLeafIter!=pLeafPointsPerProcess.end();pLeafIter++)
						{
							if(pLeafIter->first == msgRepReq.pLeafLabel)
							{
								pLeaf = reinterpret_cast<GAL_Vertex*>((pLeafIter->second).pLeaf);
								break;
							}
						}	
						assert(pLeaf);
						pRoot=pLeaf->pChild[msgRepReq.childNum];
						/*std::vector<long int>::iterator iter = pipelineStage.begin();
						for(;iter!=pipelineStage.end();iter++)
						{
							GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(*iter);
							if(pRoot->label == msgRepReq.pLeafLabel)
							{
								msgRepReq.pLeafLabel = ((GAL_PseudoRoot*)pRoot)->pLeafLabel;
								msgRepReq.pLeaf = NULL;
								msgRepReq.isLeftChild = pRoot->isLeftChild;
							}
						}	*/
					}
					assert(pRoot);
					int childNum = msgRepReq.childNum;
					msgRepSubtree.childNum = msgRepReq.childNum;
					msgRepSubtree.pLeaf = msgRepReq.pLeaf; //not used. TODO: remove this field and test.
					msgRepSubtree.pLeafLabel = msgRepReq.pLeafLabel;
					GALHelper_SaveSubtreeAsString(pRoot, msgRepSubtree.data, msgRepReq.pLeaf, childNum, false);
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
					//printf("%d received subtree data from %d\n",procRank,msg.first);
					numLocalReplicas++;
					numReplicatedVertices += msgReplicateSubtree.data.size();
					GAL_Vertex* pRoot=NULL;
					GAL_Vertex* pLeaf = NULL;
					std::vector<std::pair<int, PLeafPointBucket> >::iterator pLeafIter;
					for(pLeafIter = pLeafPointsPerProcess.begin();pLeafIter!=pLeafPointsPerProcess.end();pLeafIter++)
					{
						if(pLeafIter->first == msgReplicateSubtree.pLeafLabel)
						{
							pLeaf = reinterpret_cast<GAL_Vertex*>((pLeafIter->second).pLeaf);
							break;
						}
					}	
					assert(pLeaf);
					//printf("%d replicated subtree at %d(from %d)\n",procRank,pLeaf->label,msg.first);
					GALHelper_ReplicateSubtreeFromString(msgReplicateSubtree,&pRoot, pLeaf,msg.first);
					assert(pRoot!=NULL);
					pRoot->parent = pLeaf;
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
	GALHelper_CountSubtreeNodes(rootNode, numVerticesInRootSubtree, true);
	numReplicatedVertices +=numVerticesInRootSubtree;
	if(procRank == 0)
		printf("%d numReplicatedVertices: %ld\n",procRank,numReplicatedVertices);
	synchronize(pg);
	readyToExit=false;
	readyToExitList.clear();
}

int GAL::GALHelper_GetRandomSubtreeAndBroadcast()
{
	int ret = STATUS_SUCCESS;
	int i=0, size=numAdjacentSubtrees;
	std::vector<std::pair<long int, short int> > candidateSubtrees;
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
		std::map<long int, int>::iterator pLeafIter = pLeafMap.begin();
		for(i=0;pLeafIter!=pLeafMap.end();pLeafIter++,i++)
		{
			if(i == (*sIter/MAX_CHILDREN))
			{
				short int childNum = (*sIter%MAX_CHILDREN);
				candidateSubtrees.push_back(std::make_pair(pLeafIter->first,childNum));	
				break;
			}
		}
	}
	numReplicatedSubtrees = candidateSubtrees.size(); 
	printf("Number of replicated subtrees :%d\n",numReplicatedSubtrees);

	std::vector<std::pair<long int, short int> >::iterator subtreeIter = candidateSubtrees.begin();
	for(;subtreeIter!=candidateSubtrees.end();subtreeIter++)
	{
		GAL_Vertex* pLeaf=reinterpret_cast<GAL_Vertex*>(subtreeIter->first);
		assert(pLeaf != NULL);
		short int childNum = subtreeIter->second;
		
		if(pLeaf->pChild[childNum])
		{
			long int pRoot = reinterpret_cast<long int>(pLeaf->pChild[childNum]);
			MsgReplicateReq msg;
			msg.childNum = childNum;
			msg.pRoot = pRoot;
			msg.pLeaf = subtreeIter->first;
			msg.pLeafLabel = pLeaf->label;
			msg.numSubtreesReplicated = candidateSubtrees.size();
			GAL_SendMessage(pLeaf->childDesc[childNum],MESSAGE_REPLICATE_REQ,&msg);
			ret = STATUS_FAILURE;
		}
		else
		{
			continue;
		}
	}
	return ret;
}

int GAL::GALHelper_GetBottleneckSubtreesAndBroadcast(int numBottlenecks, char* bneckfile)
{
	int ret = STATUS_SUCCESS;
	std::vector<BneckSubtreeHdr> bottlenecks;
	std::ifstream input(bneckfile, std::fstream::in);
	if(input.fail())
	{
		printf("Bottlenecks not specified. No replication.\n");
		return ret;
	}
	else
	{
		ReadBottleneckDetails(input, numBottlenecks, bottlenecks);
	}
	numReplicatedSubtrees = bottlenecks.size(); 
	printf("Number of replicated subtrees :%d\n",numReplicatedSubtrees);

	std::vector<BneckSubtreeHdr>::iterator subtreeIter = bottlenecks.begin();
	for(;subtreeIter!=bottlenecks.end();subtreeIter++)
	{
		MsgReplicateReq msg;
		msg.pLeafLabel = subtreeIter->pLeafLabel;
		msg.childNum = subtreeIter->childNum;
		msg.pRoot = 0;
		msg.numSubtreesReplicated = bottlenecks.size();
		GAL_SendMessage(subtreeIter->procID,MESSAGE_REPLICATE_REQ,&msg);
		ret = STATUS_FAILURE;
	}
	input.close();
	return ret;
}

#endif

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

int GAL::GALHelper_TraverseUp(int step)
{
	int numPoints = allLeaves.size();
	int startIndex = procRank * (numPoints/numProcs);
	int endIndex = (procRank == (numProcs - 1))?numPoints:((procRank+1) * numPoints/numProcs);
	if(numPoints < numProcs)
	{
		if(procRank >= numPoints)
		{
			startIndex = 0;
			endIndex = 0;
		}
		else
		{
			startIndex = procRank;
			endIndex = procRank+1;
		}
	}
	

	//O(NlogN) algorithm of step 3. There also exists O(N) algorithm that is top-down. See TraverseDown.
	if(step == 3)
	{
		for(long int i=startIndex; i<endIndex;i++)
		{
			GAL_Vertex* leafNode = allLeaves[i];
			std::vector<Point>::iterator pIter = leafNode->myPoints.begin();
			for(;pIter!=leafNode->myPoints.end();pIter++)
			{
				GAL_Vertex *tmpNode = leafNode;
				while(tmpNode)
				{
					pIter->potential += tmpNode->potential;
					tmpNode = tmpNode->parent;
				}
			}
		}
	}
	else if(step == 4)
	{
		for(long int i=startIndex; i<endIndex;i++)
		{
			GAL_Vertex* leafNode = allLeaves[i];
			std::vector<Point>::iterator curNodePIter = leafNode->myPoints.begin();
			for(;curNodePIter!=leafNode->myPoints.end();curNodePIter++)
			{
				std::vector<GAL_Vertex*> neighbors;
				float x1 = (curNodePIter)->coordX;
				float y1 = (curNodePIter)->coordY;
				GALHelper_GetInteractionList(leafNode, neighbors, true);
				std::vector<GAL_Vertex*>::iterator nIter = neighbors.begin();
				for(;nIter!=neighbors.end();nIter++)
				{
					GAL_Vertex* otherNode=*nIter;
					std::vector<Point>::iterator otherNodePIter = otherNode->myPoints.begin();
					for(;otherNodePIter!=otherNode->myPoints.end();otherNodePIter++)
					{
						float x2 = (otherNodePIter)->coordX;
						float y2 = (otherNodePIter)->coordY;
						(curNodePIter)->potential += KernelFn(x1, y1, x2, y2) * (curNodePIter)->mass;
					}
				}
			}
		}
	}

	//printf("rootNode mass:%f\n",rootNode->mass);
}

int GAL::GALHelper_TraverseDown(GAL_Vertex* node, int step)
{
	std::vector<GAL_Vertex*> boxQueue;
	boxQueue.push_back(node);
	while(boxQueue.size() > 0)
	{
		GAL_Vertex* curNode = boxQueue.front();
		boxQueue.erase(boxQueue.begin());
		
		if(step == 2)
		{
			std::vector<GAL_Vertex*> wellSeparatedNodes;
			float x1 = (curNode->box.startX + curNode->box.endX) /(float) 2;
			float y1 = (curNode->box.startY + curNode->box.endY) /(float) 2;
			GALHelper_GetInteractionList(curNode, wellSeparatedNodes, false);
			std::vector<GAL_Vertex*>::iterator nIter = wellSeparatedNodes.begin();
			for(;nIter!=wellSeparatedNodes.end();nIter++)
			{
				float x2 = ((*nIter)->box.startX + (*nIter)->box.endX) /(float) 2;
				float y2 = ((*nIter)->box.startY + (*nIter)->box.endY) /(float) 2;
				curNode->potential += KernelFn(x1, y1, x2, y2) * curNode->mass;	
			}
		}

		if(curNode->isLeaf)
		{
			if(step == 3)
			{
				std::vector<Point>::iterator pIter = curNode->myPoints.begin();
				for(;pIter!=curNode->myPoints.end();pIter++)
				{
					(pIter)->potential += curNode->potential;
				}
			}
		}
		else
		{
			for(int i=0;i<MAX_CHILDREN;i++)	
			{
				if(curNode->pChild[i])
				{
				 	if(step == 3)
					{
						curNode->pChild[i]->potential += curNode->potential;
					}
					boxQueue.push_back(curNode->pChild[i]);
				}
			}
			if(boxQueue.size() > 4096)
				break;
		}
	}
	
	int numPoints = boxQueue.size();
	int startIndex = procRank * (numPoints/numProcs);
	int endIndex = (procRank == (numProcs - 1))?numPoints:((procRank+1) * (numPoints/numProcs));
	if(numPoints < numProcs)
	{
		if(procRank >= numPoints)
		{
			startIndex = 0;
			endIndex = 0;
		}
		else
		{
			startIndex = procRank;
			endIndex = procRank+1;
		}
	}
	
	GALHelper_TraverseDownParallel(boxQueue, step, startIndex, endIndex);
}

void GAL::GALHelper_TraverseDownParallel(std::vector<GAL_Vertex*>& startNodes, int step, int startIndex, int endIndex)
{
	std::vector<GAL_Vertex*> localQueue(startNodes.begin()+startIndex,startNodes.begin()+endIndex);
	while(localQueue.size() > 0)
	{
		GAL_Vertex* curNode = localQueue.front();
		localQueue.erase(localQueue.begin());
		if(step == 2)
		{
			std::vector<GAL_Vertex*> wellSeparatedNodes;
			float x1 = (curNode->box.startX + curNode->box.endX) /(float) 2;
			float y1 = (curNode->box.startY + curNode->box.endY) /(float) 2;
			GALHelper_GetInteractionList(curNode, wellSeparatedNodes, false);
			std::vector<GAL_Vertex*>::iterator nIter = wellSeparatedNodes.begin();
			for(;nIter!=wellSeparatedNodes.end();nIter++)
			{
				float x2 = ((*nIter)->box.startX + (*nIter)->box.endX) /(float) 2;
				float y2 = ((*nIter)->box.startY + (*nIter)->box.endY) /(float) 2;
				curNode->potential += KernelFn(x1, y1, x2, y2) * curNode->mass;	
			}
		}

		if(curNode->isLeaf)
		{
			if(step == 3)
			{
				std::vector<Point>::iterator pIter = curNode->myPoints.begin();
				for(;pIter!=curNode->myPoints.end();pIter++)
				{
					(pIter)->potential += curNode->potential;
				}
			}
		}
		else
		{
			for(int i=0;i<MAX_CHILDREN;i++)	
			{
				if(curNode->pChild[i])
				{
				 	if(step == 3)
					{
						curNode->pChild[i]->potential += curNode->potential;
					}
					localQueue.push_back(curNode->pChild[i]);
				}
			}
		}
	}
}

void GAL::GALHelper_GetInteractionList(GAL_Vertex* node, std::vector<GAL_Vertex*>& interactionList, bool neighbors)
{
	if(node->level < 2)
		return;
	//Get parent's neighbors and their children details to determine a node's neighbors. All other nodes (max 27) form the interactionList.
	GAL_Vertex* parent=node->parent, *gparent = node->parent->parent;
	GAL_Vertex* ggparent = (node->level>2)?gparent->parent:NULL;
	std::vector<GAL_Vertex*> pNeighbors;
	for(int i=0;i<MAX_CHILDREN;i++)
	{
		if((gparent->pChild[i]) && (gparent->pChild[i] != parent))
			pNeighbors.push_back(gparent->pChild[i]);
	}
	if(ggparent)
	{
		for(int i=0;i<MAX_CHILDREN;i++)
		{
			if(ggparent->pChild[i] && (ggparent->pChild[i] != gparent))
			{
				GAL_Vertex* gparentNeighbor = ggparent->pChild[i];
				for(int j=0;j<MAX_CHILDREN;j++)
				{
					if(gparentNeighbor->pChild[j])
					{
						if(GALHelper_AreAdjacent(gparentNeighbor->pChild[j]->box, parent->box))
							pNeighbors.push_back(gparentNeighbor->pChild[j]);
					}
				}
			}
		}
	}

	std::vector<GAL_Vertex*>::iterator nIter = pNeighbors.begin();
	for(;nIter !=pNeighbors.end(); nIter++)
	{
		for(int i=0;i<MAX_CHILDREN;i++)
		{
			if((*nIter)->pChild[i])
			{
				if(!neighbors && !GALHelper_AreAdjacent(node->box,(*nIter)->pChild[i]->box))
					interactionList.push_back((*nIter)->pChild[i]);

				else if(neighbors && GALHelper_AreAdjacent(node->box,(*nIter)->pChild[i]->box))
					interactionList.push_back((*nIter)->pChild[i]);
			}
		}
	}
}

bool GAL::GALHelper_AreAdjacent(Box& b1, Box& b2)
{
	bool ret = false;
	if((b1.startX==b2.endX) && (b1.startY == b2.startY))
		ret = true;
	else if((b1.startX==b2.endX) && (b1.endY == b2.startY))
		ret = true;
	else if((b1.startX==b2.startX) && (b1.endY == b2.startY))
		ret = true;
	else if((b1.endX==b2.startX) && (b1.endY == b2.startY))
		ret = true;
	else if((b1.endX==b2.startX) && (b1.startY == b2.startY))
		ret = true;
	else if((b1.endX==b2.startX) && (b1.startY == b2.endY))
		ret = true;
	else if((b1.startX==b2.startX) && (b1.startY == b2.endY))
		ret = true;
	else if((b1.startX==b2.endX) && (b1.startY == b2.endY))
		ret = true;
	
	return ret;
}

double GAL::KernelFn(float x1, float y1, float x2, float y2)
{
	
	return log(sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)));
}

bool GAL::GALHelper_TestContainment(GAL_Vertex* parentNode, char index)
{
	MsgBuildSubTree m(parentNode->level+1);
	std::vector<Point> childPoints;
	float sx, sy, ex, ey;
	float newnode_width = (parentNode->box.endX - parentNode->box.startX) /(float) 2;
	float newnode_height = (parentNode->box.endY - parentNode->box.startY) /(float) 2;
	newnode_width = ceil(newnode_width);
	newnode_height = ceil(newnode_height);
	assert(newnode_width == newnode_height);

	switch(index)
	{
		case 1:
		      {
			sx = parentNode->box.startX;
			sy = parentNode->box.startY;
		      }
		      break;
		case 0:
		      {
			sx = parentNode->box.startX + newnode_width;
			sy = parentNode->box.startY;
		      }
		      break;
		case 2:
		      {
			sx = parentNode->box.startX;
			sy = parentNode->box.startY + newnode_height ;
		      }
		      break;
		case 3:
		      {
			sx = parentNode->box.startX + newnode_width ;
			sy = parentNode->box.startY + newnode_height ;
		      }
		      break;
	}
	ex = sx +  newnode_width ;
	ey = sy + newnode_height ;
	std::vector<Point>::iterator iter = parentNode->myPoints.begin();
	for (;iter!=parentNode->myPoints.end();) 
	{
		if (((iter)->coordX >= sx)  && ((iter)->coordX <= ex) && ((iter)->coordY >= sy)  && ((iter)->coordY <= ey))
		{
			childPoints.push_back(*iter);
			iter = parentNode->myPoints.erase(iter); 
		}
		else
			++iter;
	}

	Box childBox(sx,ex,sy,ey);
	m.pLeaf.push_back(std::make_pair(procRank,reinterpret_cast<long int>(parentNode)));
	m.childNo = index;
	m.box = childBox;
#ifdef SPAD_2
	m.pLeafLabel = parentNode->label;
#endif
	m.numPoints = childPoints.size();
	if(m.numPoints == 0)
		return false;

	parentNode->numChildren++;
#ifdef MERGE_HEIGHT1_TREES
	if(m.numPoints <= NUM_POINTS_PER_CELL)
	{
		GAL_Vertex* intNode = GAL_CreateVertex(VTYPE_PSEUDOROOT);
		intNode->pseudoRoot = true; //a leaf that is named as pseudoRoot for implementation compatibility.
		intNode->isLeaf = true;
		allLeaves.push_back(intNode);	
		parentNode->pChild[index] = intNode;
		parentNode->childDesc[index] = procRank;
		intNode->level = parentNode->level + 1;
		intNode->parentDesc = procRank;
		intNode->parent = parentNode;
		intNode->box = childBox;
		if(depth < intNode->level)
			depth  = intNode->level;
		for(int i=0;i<childPoints.size();i++)
		{
			intNode->myPoints.push_back(childPoints[i]);
			intNode->mass += childPoints[i].mass;
		}
		return false;
	}
#endif
	if(procRank != 0)
	{
		return true;
	}
	numExpectedUpdates++;
	parentNode->pseudoLeaf = true;
	if(m.numPoints < 500)
	{
		int nextprocessid = GALHelper_GetNextProcessId(reinterpret_cast<long int>(parentNode));
		for(int i=0;i<childPoints.size();i++)
		{
			m.ptv.push_back(childPoints[i]);
		}
		GAL_SendMessage(nextprocessid,MESSAGE_BUILDSUBTREE,&m); 
		return true;
	}
	else 
	{
		int nextprocessid = GALHelper_GetNextProcessId(reinterpret_cast<long int>(parentNode));
		int j = 0;
		while(true)
		{
			int count=0;
			long int i=0;
			m.hasMoreData=true;
			m.ptv.clear();
			for(i=j;i<m.numPoints;i++, count++)
			{
				if(count == 500)
				{
					j=i;
					break;
				}
				m.ptv.push_back(childPoints[i]);
			}
			if(i >= m.numPoints)
			{
				m.hasMoreData=false;
			}
			GAL_SendMessage(nextprocessid,MESSAGE_BUILDSUBTREE,&m); 
			if(!(m.hasMoreData))
				break;
		}
		return true;
	}
}


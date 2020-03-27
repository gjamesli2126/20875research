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

#ifdef DYNAMIC_LB
extern int DYNAMIC_LB_TRIGGER;
extern int STATIC_LB_TRIGGER;
std::map<GAL_Vertex*,int> replicatedSubtreeRequests;
std::map<GAL_Vertex*, std::vector<std::string> > subtreeCache;
int numReplicas;
int** bneckList=NULL;
#define NUMB 7
#endif


typedef struct PLeafPointBucket{
int depth;
bool isLeft;
long int pLeaf;
long int from;
long int to;
long int totalPts;
}PLeafPointBucket;

std::map<int, PLeafPointBucket> pLeafPointsPerProcess;

std::map<GAL_Vertex*,int> replicatedSubtreeTable;
MsgUpdatePLeaves msgUpdatePLeaves;
long int labelCount; //counter to assign label values to a node.
int sort_split;
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

GAL* GAL::instance = NULL;

int compare_point(const void *a, const void *b) {
	switch(sort_split)
	{
		case 0:
			if(((Point *)a)->cofm.x < ((Point *)b)->cofm.x) {
				return -1;
			} else if(((Point *)a)->cofm.x > ((Point *)b)->cofm.x) {
				return 1;
			} else {
				return 0;
			}
			break;
		case 1:
			if(((Point *)a)->cofm.y < ((Point *)b)->cofm.y) {
				return -1;
			} else if(((Point *)a)->cofm.y > ((Point *)b)->cofm.y) {
				return 1;
			} else {
				return 0;
			}
			break;
		case 2:
			if(((Point *)a)->cofm.z < ((Point *)b)->cofm.z) {
				return -1;
			} else if(((Point *)a)->cofm.z > ((Point *)b)->cofm.z) {
				return 1;
			} else {
				return 0;
			}
			break;
	}

}

float mydistance(Vec& a, Vec& b) {
	float d = 0;
	
	float  tmp = (a.x - b.x);
	float tmpsq = tmp * tmp;
	d +=tmpsq;
	tmp = (a.y - b.y);
	tmpsq = tmp * tmp;
	d +=tmpsq;
	tmp = (a.z - b.z);
	tmpsq = tmp * tmp;
	d +=tmpsq;
	return sqrt(d);
}

/* based on clrs, chap 7,9.*/
int partition(Point* a, int p, int r, void* comparator)
{
	Point x = a[r];
	int i=p-1;
	for(int j=p;j<=r-1;j++)
	{
		if(compare_point(&(a[j]), &x) == -1)
		{
			i=i+1;
			std::swap(a[i],a[j]);
		}
	}
	std::swap(a[i+1],a[r]);
	return i+1;
}

void find_median(Point* a, int p, int r, int i, void* comparator)
{
	if((p==r)||(p>r))
		return;
	int q = partition(a, p, r, comparator);
	int k = q-p+1;
	if(k==i)
		return; 
	else if(i<k)
		return find_median(a,p,q-1,i, comparator);
	else
		return find_median(a,q+1,r,i-k, comparator);
			
}

void my_nth_element(Point* points, int from, int mid, int to,void* comparator)
{
	find_median(points, from, to, mid, comparator); 
}


void compute_cell_params(Point* points, long int lb, long int ub, Vec& cofm, float& mass, float& radius)
{
	float min[DIMENSION],max[DIMENSION];
	for(int i=0;i<DIMENSION;i++)
	{
		min[i]=FLT_MAX;
		max[i]=-FLT_MAX;
	}
	for (long int i = lb; i <= ub; i++) 
	{
		/* compute cofm */
		mass += points[i].mass;
		cofm.x += points[i].mass * points[i].cofm.x;
		cofm.y += points[i].mass * points[i].cofm.y;
		cofm.z += points[i].mass * points[i].cofm.z;
		/* compute bounding box */
		if(min[0] > points[i].cofm.x)
			min[0] = points[i].cofm.x;
		if(min[1] > points[i].cofm.y)
			min[1] = points[i].cofm.y;
		if(min[2] > points[i].cofm.z)
			min[2] = points[i].cofm.z;
		if(max[0] < points[i].cofm.x)
			max[0] = points[i].cofm.x;
		if(max[1] < points[i].cofm.y)
			max[1] = points[i].cofm.y;
		if(max[2] < points[i].cofm.z)
			max[2] = points[i].cofm.z;
	}
	/* compute final cofm */
	cofm.x /= mass;
	cofm.y /= mass;
	cofm.z /= mass;

	/* compute center of the box */
	Vec center(0.);
	for(int i=0;i<DIMENSION;i++)
	{
		float coord_i = (min[i]+max[i])/(float)2;
		if(i==0)
			center.x = coord_i;	
		if(i==1)
			center.y = coord_i;	
		else
			center.z = coord_i;	
	}
	
	/* compute Bmax and Bcel */
	float Bmax=0.,Bcel=0.;
	for (long int i = lb; i <= ub; i++) 
	{
		float bmax = mydistance(cofm, points[i].cofm);
		if(Bmax < bmax)
			Bmax = bmax;
		float bcel = mydistance(center, points[i].cofm);
		if(Bcel < bcel)
			Bcel = bcel;
	}
	/* compute ropen aka radius of cell
	formula based on the PKDGRAV paper: http://hpcc.astro.washington.edu/faculty/trq/brandon/pkdgrav.html */
	//TODO: pass tol to TAL from Application layer. Currently harcoded to 0.5 (theta parameter).
	radius = Bmax/(sqrt(3) * 0.5) + Bcel;	

}



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
		case MESSAGE_POINTBUCKET:
					send_oob(pg,processId,messageId,*(reinterpret_cast<MsgPointBucket *>(msg))); 
					break;
		case MESSAGE_NEWBUCKETREQ:
					send_oob(pg,processId,messageId,*(reinterpret_cast<int *>(msg))); 
					break;
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
#ifdef DYNAMIC_LB
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
#endif
}

#ifdef HYBRID_BUILD
void GAL::DetermineBoundingBox(Point* points,long int from, long int to, float* globalMin, float* globalMax)
{
		float *localMins = new float[DIMENSION]; 	
		float *localMaxs = new float[DIMENSION]; 	
		for(int i=0;i<DIMENSION;i++)
		{
			localMins[i]=FLT_MAX;
			localMaxs[i]=-FLT_MAX;
		}
		/*for(long int i=from;i<=to;i++)
		{
			for(long int j=0;j<DIMENSION;j++)
			{
				if(points[i].pt[j] < localMins[j])
					localMins[j] = points[i].pt[j];
				if(points[i].pt[j] > localMaxs[j])
					localMaxs[j] = points[i].pt[j];
			}
		}*/
		for(int i=0;i<DIMENSION;i++)
		{
			reduce(communicator(pg), localMins[i],globalMin[i],boost::parallel::minimum<float>(),0);
			broadcast(communicator(pg), globalMin[i], 0);
			reduce(communicator(pg), localMaxs[i],globalMax[i],boost::parallel::maximum<float>(),0);
			broadcast(communicator(pg), globalMax[i], 0);
		}
		delete [] localMins;
		delete [] localMaxs;

}

GAL_Vertex* GAL::Recurse(Point* points, GAL_Vertex* parent, long int from, long int to, int tmpHeight, int axis, bool isLeftChild, int& leafnodeId)
{
	int mid=-1;
	GAL_Vertex* node=NULL, *leftnode=NULL, *rightnode=NULL;
	float globalMin[DIMENSION], globalMax[DIMENSION];
	/*printf("from:%d to %d\n",from, to);
	if((from==3983) && (to==3983))
		printf("break\n");*/
	if((tmpHeight==subtreeHeight) && (numProcs != 1))
	{
		long int localNumPoints = to-from+1, globalNumPoints=0;
		if(from > to)
			localNumPoints = 0;
		reduce(communicator(pg), localNumPoints,globalNumPoints,std::plus<long int>(),0);
		broadcast(communicator(pg),globalNumPoints,0);
		/*if(procRank == 0)
			printf("%d points in leafnode id %d\n", globalNumPoints,++leafnodeId); */
		++leafnodeId;
		if(globalNumPoints > 1)
		{
			PLeafPointBucket bucket;
			bucket.totalPts=globalNumPoints;
			bucket.pLeaf=reinterpret_cast<long int>(parent);
			bucket.from = from;	
			bucket.to=to;
			bucket.depth=tmpHeight;
			bucket.isLeft=isLeftChild;
			pLeafPointsPerProcess.insert(std::make_pair(leafnodeId, bucket));
		}
		else if(globalNumPoints == 1)
		{
			DetermineBoundingBox(points,from,to, globalMin, globalMax);
			GAL_Vertex* leafNode;
			leafNode = GAL_CreateVertex(VTYPE_NORMAL);
			(*g)[leafNode->desc].uCount = 2;
			leafNode->cp.cofm.x = globalMin[0];
			leafNode->cp.cofm.y = globalMin[1];
			leafNode->cp.cofm.z = globalMin[2];
			//TODO:leafNode->lp = new LeafParams(points[from].vel,points[from].acc);
			/*leafNode->pt = new PointCoord();
			for(int i=0;i<DIMENSION;i++)
			{
				leafNode->pt->coord_d[i] = globalMin[i];
				if(leafNode->max_d[i] < globalMax[i])
					leafNode->max_d[i] = globalMax[i];
				if(leafNode->min_d[i] > globalMin[i])
					leafNode->min_d[i] = globalMin[i];

			}*/
			leafNode->isLeftChild = isLeftChild;
			leafNode->level = tmpHeight;
	#ifdef TRAVERSAL_PROFILE
			std::pair<std::map<int,long int>::iterator,bool> ret;
			ret = numLeavesAtHeight.insert(std::pair<int,long int>(tmpHeight,1) );
			if(ret.second==false)
				numLeavesAtHeight[tmpHeight] += 1;
	#endif
			return leafNode;

		}
		
		return NULL;
	}

	DetermineBoundingBox(points,from,to, globalMin, globalMax);
	long int localNumPoints = to-from+1, globalNumPoints;
	if(from > to)
		localNumPoints=0;
	reduce(communicator(pg), localNumPoints,globalNumPoints,std::plus<long int>(),0);
	broadcast(communicator(pg), globalNumPoints, 0);
	if(globalNumPoints == 1)
	{
		GAL_Vertex* leafNode;
		leafNode = GAL_CreateVertex(VTYPE_NORMAL);
		(*g)[leafNode->desc].uCount = 2;
		leafNode->cp.cofm.x = globalMin[0];
		leafNode->cp.cofm.y = globalMin[1];
		leafNode->cp.cofm.z = globalMin[2];
		//TODO: leafNode->lp = new LeafParams(points[from].vel,points[from].acc);
		/*leafNode->pt = new PointCoord();
    		for(int i=0;i<DIMENSION;i++)
		{
			leafNode->pt->coord_d[i] = globalMin[i];
			if(leafNode->max_d[i] < globalMax[i])
				leafNode->max_d[i] = globalMax[i];
			if(leafNode->min_d[i] > globalMin[i])
				leafNode->min_d[i] = globalMin[i];

		}*/
		leafNode->isLeftChild = isLeftChild;
		leafNode->level = tmpHeight;
#ifdef TRAVERSAL_PROFILE
		std::pair<std::map<int,long int>::iterator,bool> ret;
		ret = numLeavesAtHeight.insert(std::pair<int,long int>(tmpHeight,1) );
		if(ret.second==false)
			numLeavesAtHeight[tmpHeight] += 1;
#endif
		return leafNode;
	}
	if(globalNumPoints == 0)
	{
		return NULL;
	}
	node = GAL_CreateVertex(VTYPE_NORMAL);
	float median=(globalMin[axis]+globalMax[axis])/(float)2;
	mid = DeterminePartitionElements(points, from, to, median, axis);
	/*if(mid == (from-1))
		mid = from;*/
	leftnode = Recurse(points, node, from, mid, tmpHeight+1, (axis+1)%DIMENSION, true, leafnodeId);
	rightnode = Recurse(points, node, mid+1, to, tmpHeight+1, (axis+1)%DIMENSION, false, leafnodeId);
	node->level = tmpHeight;
	node->leftChild=leftnode;
	node->rightChild=rightnode;
	node->leftDesc=node->rightDesc=procRank;
	if(node->leftChild)
	{
		(node->leftChild)->parent = node;
		(node->leftChild)->parentDesc = procRank;
	}
	if(node->rightChild)
	{
		node->rightChild->parent = node;
		(node->rightChild)->parentDesc = procRank;
	}
	return node;
}

long int GAL::DeterminePartitionElements(Point* points, long int from, long int to, float median, int axis)
{
	if(from > to)
		return from;
	sort_split = axis;
	qsort(&points[from], to - from + 1, sizeof(Point), compare_point);
	long int i;
	if(median == points[from].pt[axis])
		return from;
	for(i=from;i<=to;i++)
	{
		if(points[i].pt[axis] < median)
			continue;
		else
			break;
	}
	return i-1;
}

#endif



int GAL::GAL_ConstructKDTree(Point* points, int numPoints, bool hasMoreData, int stHeight)
{
#ifdef PARALLEL_BGL
	bool done = false;
	int depth = 0;
	subtreeHeight = stHeight;
	int nxtPtTBInserted, curLeafId;
	static int batchNumber=0;
	long int curPointIndx=0, prevPointIndx=-1;
	bool replicatedTopSubtree=false;
	bool curBucketSent = false;

	if(batchNumber > 0)
		nxtPtTBInserted=-1;//(numPoints-1)/2;
	else
		nxtPtTBInserted=(numPoints-1);
		
	batchNumber++;

	depth = 0;
	
	MsgBuildSubTree msgBuildSubTree, msgSubtreeSecondLevel;
	MsgUpdateMinMax msgDoneTree;
	
	int donebuildsubtree;
	/*if (numProcs==1)
		subtreeHeight=-1;*/
	CellParams cpar;

#ifdef HYBRID_BUILD
	
	int status=STATUS_SUCCESS;
	if (numProcs==1)
		done=true;
	int leafnodeId=0;
	rootNode = Recurse(points, NULL, 0, numPoints-1, 0, 0, false, leafnodeId);
	long int tPPP=0;
	std::map<int, PLeafPointBucket>::iterator mIter = pLeafPointsPerProcess.begin();
	curLeafId = mIter->first;
	if(pLeafPointsPerProcess.size() == 0)
	{
		printf("Data parallel tree formation.\n");
		done =true;
	}
	
#else
		::compute_cell_params(points, 0, nxtPtTBInserted, cpar.cofm, cpar.mass, cpar.ropen);
		donebuildsubtree = GAL_BuildSubTree(points,NULL, 0, nxtPtTBInserted, 0, 0, false, cpar);
		if(donebuildsubtree == BUILDSUBTREE_SAMEPROCESS)
				done=true;
#endif
	
	PBGL_oobMessage msg;
	int donetreeval;
	while (!done)
	{
		//poll for messages
		PBGL_AsyncMsg pollMsg = pg.poll();
		if(!pollMsg)
		{
#ifdef HYBRID_BUILD
			if((procRank != 0) && !curBucketSent)
			{
				curLeafId = mIter->first;
				curBucketSent = true;
				//leafNodeId, isLeftChild, depth, pseudoLeaf pointer
				MsgPointBucket m(mIter->first,(mIter->second).isLeft,(mIter->second).depth,(mIter->second).pLeaf);
				if(((mIter->second).to - (mIter->second).from+1) > 5000)
				{
					int j=(mIter->second).from;
					while(true)
					{
						int count=0;
						long int i=0;
						m.hasMoreData=true;
						m.ptv.clear();
						for(i=j;i<=(mIter->second).to;i++, count++)
						{
							if(count == 5000)
							{
								j=i;
								break;
							}
							m.ptv.push_back(points[i]);
						}
						if(i > (mIter->second).to)
						{
							m.hasMoreData=false;
						}
						GAL_SendMessage(0,MESSAGE_POINTBUCKET,&m); 
						if(!(m.hasMoreData))
							break;
					}
				}
				else
				{
					for(long int i=(mIter->second).from;i<=(mIter->second).to;i++)
					{
						m.ptv.push_back(points[i]);
					}
					GAL_SendMessage(0,MESSAGE_POINTBUCKET,&m); 
				}

			}
#endif
			continue;
		}
		else
			msg = pollMsg.get();

		switch(msg.second)
		{
#ifdef HYBRID_BUILD
			case MESSAGE_POINTBUCKET:
				{
					MsgPointBucket msgPointBucket;
					receive_oob(pg, msg.first, msg.second, msgPointBucket);
					if((mIter->second).totalPts > 5000)
					{
						msgSubtreeSecondLevel.numPoints+=msgPointBucket.ptv.size();
						std::ofstream pointBucketOutput;
						pointBucketOutput.open("PointBucketFile.txt", std::ios_base::app);
						WritePoints(pointBucketOutput, msg.first, &(msgPointBucket.ptv), 0, msgPointBucket.ptv.size()-1);
						pointBucketOutput.close();
					}
					else
					{
						msgSubtreeSecondLevel.ptv.insert(msgSubtreeSecondLevel.ptv.end(),msgPointBucket.ptv.begin(),msgPointBucket.ptv.end());
					}
						
					assert(curLeafId == msgPointBucket.pLeafId);
					//	printf("Error curLeafId %d msgPointBucketId:%d\n", curLeafId, msgPointBucket.pLeafId);
					if(!(msgPointBucket.hasMoreData))
					{
						msgSubtreeSecondLevel.depth = msgPointBucket.depth;
						msgSubtreeSecondLevel.pLeaf.push_back(std::make_pair(msg.first,msgPointBucket.pLeaf));
						msgSubtreeSecondLevel.isleft = msgPointBucket.isLeft;
					}
					if(msgSubtreeSecondLevel.pLeaf.size() == (numProcs-1))
					{
						if((mIter->second).totalPts > 5000)
						{
							msgSubtreeSecondLevel.numPoints+=(mIter->second).to - (mIter->second).from + 1;
							std::ofstream pointBucketOutput;
							pointBucketOutput.open("PointBucketFile.txt", std::ios_base::app);
							WritePoints(pointBucketOutput,procRank, points,(mIter->second).from, (mIter->second).to);
							pointBucketOutput.close();
							//printf("leafNode %d points size %ld\n",curLeafId,msgSubtreeSecondLevel.numPoints); 
						}
						else
						{
							for(long int i=(mIter->second).from;i<=(mIter->second).to;i++)
								msgSubtreeSecondLevel.ptv.push_back(points[i]);
							//printf("leafNode %d points size %ld\n",curLeafId,msgSubtreeSecondLevel.ptv.size()); 
						}

						msgSubtreeSecondLevel.pLeaf.push_back(std::make_pair(procRank,(mIter->second).pLeaf));
						msgSubtreeSecondLevel.subroot = (reinterpret_cast<GAL_Vertex*>((mIter->second).pLeaf))->desc;
						int nextprocessid = GALHelper_GetNextProcessId((mIter->second).pLeaf);
						GAL_SendMessage(nextprocessid,MESSAGE_BUILDSUBTREE,&msgSubtreeSecondLevel); 
						msgSubtreeSecondLevel.pLeaf.clear();
						msgSubtreeSecondLevel.numPoints=0;
						msgSubtreeSecondLevel.ptv.clear();
					}
				}
				break;
				case MESSAGE_NEWBUCKETREQ:
				{
					int recvVal;
				     	receive_oob(pg, msg.first, msg.second, recvVal);
					curBucketSent=false;
					if(mIter != pLeafPointsPerProcess.end())
					{
						mIter++;
					}
		
				}
				break;
#endif
			case MESSAGE_BUILDSUBTREE: 
				{
				     	receive_oob(pg, msg.first, msg.second, msgBuildSubTree);
					GAL_Vertex node;
					GAL_Vertex* tmpLeaf;
#ifdef TRAVERSAL_PROFILE
					numLinks++;
#endif
					node.desc = msgBuildSubTree.subroot;
					node.parent = reinterpret_cast<GAL_Vertex*>(msgBuildSubTree.pLeaf[msgBuildSubTree.pLeaf.size()-1].second);
				
					if((node.desc).owner !=  procRank)
						tmpLeaf = &node;
					else
						tmpLeaf = node.parent;

					int size=0;
					if(msgBuildSubTree.ptv.size() != 0)
						size = msgBuildSubTree.ptv.size();
					else
						size = msgBuildSubTree.numPoints;

					Point* pointArr = new Point[size];	
					if(msgBuildSubTree.ptv.size() != 0)
					{
						for(int i=0;i<msgBuildSubTree.ptv.size();i++)
							pointArr[i] = msgBuildSubTree.ptv[i];
					}
					else
					{
#ifdef HYBRID_BUILD
						std::ifstream input("PointBucketFile.txt", std::fstream::in);
						if(input.fail())
						{
							std::cout<<"File does not exist. exiting"<<std::endl;
							//TODO: Do this cleanly. Send message to all processes to exit.
							MPI_Finalize();
							exit(0);
						}
						ReadPoints(input, 0, size, pointArr);
						remove("PointBucketFile.txt");
						input.close();
#else
						for(int i=msgBuildSubTree.from;i<=msgBuildSubTree.to;i++)
							memcpy(&pointArr[i], &points[i], sizeof(Point));
#endif
					}

					donebuildsubtree = GAL_BuildSubTree(pointArr,tmpLeaf, 0, size-1, msgBuildSubTree.depth, 0, msgBuildSubTree.isleft, msgBuildSubTree.cpar);
					delete [] pointArr;
					GAL_Vertex* tmpNode = NULL;
					if(msgBuildSubTree.isleft)
						tmpNode = tmpLeaf->leftChild;
					else
						tmpNode = tmpLeaf->rightChild;
					if(msgBuildSubTree.depth == subtreeHeight)
					{
						assert(tmpNode->pseudoRoot);
						//((GAL_PseudoRoot*)tmpNode)->parents = msgBuildSubTree.pLeaf;
						std::vector<std::pair<int, long int> >::iterator pLeafIter = msgBuildSubTree.pLeaf.begin();
						for(;pLeafIter!=msgBuildSubTree.pLeaf.end();pLeafIter++)
							((GAL_PseudoRoot*)tmpNode)->parents2.insert(*pLeafIter);
					}

					//If all the vertices of the subtree are owned by a single process, inform the caller process that subtree construction is done.
					if(donebuildsubtree == BUILDSUBTREE_SAMEPROCESS)
					{
						//construct donesubtree message to notify parent
						msgDoneTree.parent = (msgBuildSubTree.subroot).owner;
						if(msgBuildSubTree.isleft)
						{
							msgDoneTree.pRoot = reinterpret_cast<long int>(tmpLeaf->leftChild);
						}
						else
						{
							msgDoneTree.pRoot = reinterpret_cast<long int>(tmpLeaf->rightChild);
						}
				
						msgDoneTree.pLeaf = reinterpret_cast<long int>(node.parent);
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
				
				if(vert->level == (subtreeHeight-1))
				{
					MsgUpdatePLeaf msgUpdatePLeaf;
#ifdef HYBRID_BUILD
					msgUpdatePLeaf.label = curLeafId;
					mIter++;
					if(mIter != pLeafPointsPerProcess.end())
					{
						/*if((curLeafId % 1000) == 0)
							printf("leaf %d done. next leaf %d\n", curLeafId, mIter->first);*/
						curBucketSent=false;
						int msgVal = curLeafId;
						for(int i=1;i<numProcs;i++)
							GAL_SendMessage(i,MESSAGE_NEWBUCKETREQ,&msgVal);
						curLeafId = mIter->first;
					}
					else
					{
						done=true;
					}
#else
					msgUpdatePLeaf.label = vert->label;
#endif

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
					//msgUpdatePLeaf.leafDesc = tmpVert->desc;
					vPLeaves.push_back(msgUpdatePLeaf);						
				}

				//repeatedly update all the nodes up the tree if update count of any of those is 2.
				if((*g)[tmpVert->desc].uCount == 2)
				{
					while(tmpVert)
					{
						//check if already at root node. If so initiate termination
						if(tmpVert->parent == NULL)
						{
#ifndef HYBRID_BUILD
							done =true;
#endif
							break;
						}
						if(!tmpVert->pseudoRoot)
						{
							tmpVert = tmpVert->parent;
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
							msgDoneTree.parent = tmpVert->parentDesc;
							msgDoneTree.isLeft = tmpVert->isLeftChild;
							msgDoneTree.pRoot = reinterpret_cast<long int>(tmpVert);
							msgDoneTree.pLeaf = reinterpret_cast<long int>(tmpVert->parent);
							GAL_SendMessage((tmpVert->parentDesc),MESSAGE_DONESUBTREE,&msgDoneTree);
							break;	
						}
					}		
				}

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
						if(procRank==1)printf("%d NUM_PSEUDOLEAVES %d \n",procRank,msgUpdatePLeaves.vPLeaves.size());
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
	/*synchronize(pg);
	if(procRank ==0)
		remove("ReplicatedTree.txt");*/
#ifdef DYNAMIC_LB
	synchronize(pg);
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


#endif
	//printf("LabelCount:%d\n",labelCount);
	/*FILE* fp = fopen("treelog2.txt","w+");
	print_treetofile(fp);
	fclose(fp);*/

	
return 0;
}

void GAL::GALHelper_DeleteSubtree(GAL_Vertex* node)
{
		assert(node != NULL);

		assert((node->desc).owner ==procRank); 
		
		if((node->leftChild==NULL) && (node->rightChild==NULL))
		{
#ifdef TRAVERSAL_PROFILE
			numLeavesAtHeight[node->level] -= 1;
			if(numLeavesAtHeight[node->level] < 0)
				numLeavesAtHeight[node->level] = 0;
#endif
			GAL_DeleteVertex(node);
			return;
		}
		
		if((node->leftDesc == procRank) && node->leftChild)
			GALHelper_DeleteSubtree(node->leftChild);

		if((node->rightDesc == procRank) && (node->rightChild))
			GALHelper_DeleteSubtree(node->rightChild);

		GAL_DeleteVertex(node);
}

void GAL::GAL_DeleteVertex(GAL_Vertex* node)
{
	remove_vertex(node->desc, *g);
	delete node;
}

/* Returns the parent(s) of a vertex. In case of parallel BGL, since an adjacent node can be owned by different process, it is not known whether that vertex is a parent or not. Hence It is upto the caller of this API to determine the status. However, the vector parentVert also contains the list of all such adjacent vertices.*/
//TODO: remove numParents once a working code is ready

/*IMP: Caller of this function must delete the vertex type returned after its usage */

void* GAL::GAL_GetVertexProperties(GAL_Vertex& v, int vertextype)
{
#ifdef PARALLEL_BGL
	BGLTreeNode *ret = NULL;
	switch(vertextype)
	{
		case KDTREE: 
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
		case KDTREE: 
			(*g)[v.desc].uCount = reinterpret_cast<BGLTreeNode*>(refnode)->uCount;
			break;


	}
#endif
}

int GAL::GAL_BuildSubTree(Point* points, GAL_Vertex* subtreeRoot,int from, int to, int height, int DOR, bool isLeft, CellParams& cpar)
{
	if(to < from)
	{
		return BUILDSUBTREE_FAILURE;
	}
	//If the subtree of configured height is created, ask a different process to take over the creation of subtrees that are down the hierarchy
#ifdef MERGE_HEIGHT1_TREES
	else if((numProcs> 1) && (DOR == subtreeHeight) && ((to - from) != 0))
#else
	else if((numProcs > 1) && (DOR == subtreeHeight))
#endif
	{
#ifndef HYBRID_BUILD
			PLeafPointBucket bucket;
			bucket.pLeaf=reinterpret_cast<long int>(subtreeRoot);
			bucket.from = from;	
			bucket.to=to;
			bucket.depth=height;
			bucket.isLeft=isLeft;
			pLeafPointsPerProcess.insert(std::make_pair(subtreeRoot->label, bucket));
#endif

		subtreeRoot->pseudoLeaf = true;
		if((procRank != 0) && (height == subtreeHeight))
		{
			return BUILDSUBTREE_FAILURE;
		}
		MsgBuildSubTree m(isLeft,height,subtreeRoot->desc);
		m.pLeaf.push_back(std::make_pair(procRank,reinterpret_cast<long int>(subtreeRoot)));
		m.cpar = cpar;
#ifdef HYBRID_BUILD
		if((to - from+1) > 5000)
			printf("Warning. MPI buffer might overflow\n");
#else
		m.from=from;
		m.to=to;
		m.numPoints = to-from+1;
		if((to - from+1) < 5000)
#endif
		{
			for(int i=from;i<=to;i++)
			{
				m.ptv.push_back(points[i]);
			}
		}

		//get next process
		int nextprocessid = GALHelper_GetNextProcessId(reinterpret_cast<long int>(subtreeRoot));
		GAL_SendMessage(nextprocessid,MESSAGE_BUILDSUBTREE,&m); 
		return BUILDSUBTREE_FAILURE;
	}
	else if(from == to)
	{
		//Create leaf vertex and set properties
		GAL_Vertex* leafNode;
		if((DOR == 0) || (DOR == subtreeHeight))
		{
			leafNode = GAL_CreateVertex(VTYPE_PSEUDOROOT);
			leafNode->pseudoRoot = true;
		}
		else	
			leafNode = GAL_CreateVertex(VTYPE_NORMAL);
		(*g)[leafNode->desc].uCount = 2;
		leafNode->isLeftChild = isLeft;
		leafNode->parent = subtreeRoot;
		leafNode->parentDesc = (subtreeRoot->desc).owner;
		leafNode->level = height;
		leafNode->cp  = cpar;
		leafNode->lp = new LeafParams(points[from].vel,points[from].acc);
		
#ifdef TRAVERSAL_PROFILE
		std::pair<std::map<int,long int>::iterator,bool> ret;
		ret = numLeavesAtHeight.insert(std::pair<int,long int>(height,1) );
		if(ret.second==false)
			numLeavesAtHeight[height] += 1;
#endif
		//If parent belongs to a different node, Send message to parent that subtree construction is done.
		if((subtreeRoot->desc).owner != (leafNode->desc).owner)
		{
			leafNode->parentDesc = (subtreeRoot->desc).owner;
			assert(leafNode->parentDesc==0);
			leafNode->parent = subtreeRoot->parent;

			MsgUpdateMinMax msgDoneTree;
			msgDoneTree.parent = (subtreeRoot->desc).owner;
			msgDoneTree.pRoot = reinterpret_cast<long int>(leafNode);
			msgDoneTree.pLeaf = reinterpret_cast<long int>(subtreeRoot->parent);
			msgDoneTree.isLeft = isLeft;
			GAL_SendMessage((subtreeRoot->desc).owner,MESSAGE_DONESUBTREE,&msgDoneTree);
#ifdef TRAVERSAL_PROFILE
			numPRootLeaves++;
#endif

			return BUILDSUBTREE_FAILURE;//SENTOOB;
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
			
			return BUILDSUBTREE_SAMEPROCESS;
		}
	}
	else
	{
		//reached int node 
		int leftstatus,rightstatus,status=BUILDSUBTREE_FAILURE;
		int oobsent = BUILDSUBTREE_SAMEPROCESS;
		int flag =BUILDSUBTREE_SAMEPROCESS;
		int mid = (from + to) / 2;
		//create a vertex for the intermediate node and add it to the graph
		GAL_Vertex* intNode; 
		if((DOR == 0)||(DOR==subtreeHeight))
		//if((DOR == 0)||(DOR==11))
			intNode = GAL_CreateVertex(VTYPE_PSEUDOROOT);
		else	
			intNode = GAL_CreateVertex(VTYPE_NORMAL);

		intNode->level = height;
		intNode->isLeftChild = isLeft;
		intNode->cp  = cpar;
		
		float min[DIMENSION],max[DIMENSION];
		for(int i=0;i<DIMENSION;i++)
		{
			min[i]=FLT_MAX;
			max[i]=-FLT_MAX;
		}
		for (long int i = from; i <= to; i++) 
		{
			/* compute bounding box */
			if(min[0] > points[i].cofm.x)
				min[0] = points[i].cofm.x;
			if(min[1] > points[i].cofm.y)
				min[1] = points[i].cofm.y;
			if(min[2] > points[i].cofm.z)
				min[2] = points[i].cofm.z;
			if(max[0] < points[i].cofm.x)
				max[0] = points[i].cofm.x;
			if(max[1] < points[i].cofm.y)
				max[1] = points[i].cofm.y;
			if(max[2] < points[i].cofm.z)
				max[2] = points[i].cofm.z;
		}
		float maxWidth=max[0]-min[0];
		sort_split=0;
		for(int i=1;i<DIMENSION;i++)
		{
			if((max[i]-min[i]) > maxWidth)
			{
				maxWidth = max[i] - min[i];
				sort_split=i;
			}
		}
		//qsort(&points[from], to - from + 1, sizeof(Point), compare_point);

		int dummy;
		::my_nth_element(points, from, (to-from)/2, to,&dummy);


		CellParams leftCell, rightCell; 
		::compute_cell_params(points, from, mid, leftCell.cofm, leftCell.mass, leftCell.ropen);
		::compute_cell_params(points, mid+1, to, rightCell.cofm, rightCell.mass, rightCell.ropen);

		
		if(subtreeRoot != NULL)
		{
			intNode->isLeftChild = isLeft;
			//set parent descriptor
			intNode->parentDesc = (subtreeRoot->desc).owner;
			//Create edge between intermediate node and its parent and set properties
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
		leftstatus = GAL_BuildSubTree(points,intNode,from, mid,height+1, DOR+1, true, leftCell);

		//We need to update the parent of the subtree rooted at intNode that left subtree construction is done or not done.
		if(leftstatus != BUILDSUBTREE_FAILURE)
		{
				//update intNode bounding box
				GAL_Vertex* tmpNode = intNode->leftChild;
				(*g)[intNode->desc].uCount++;
		}

		//build right tree
		rightstatus = GAL_BuildSubTree(points,intNode, mid+1, to,height+1, DOR+1, false, rightCell);
		
		if(rightstatus != BUILDSUBTREE_FAILURE)
		{
					//update intNode bounding box
				GAL_Vertex* tmpNode = intNode->rightChild;
				(*g)[intNode->desc].uCount++;

		}
		if((leftstatus != BUILDSUBTREE_FAILURE) && (rightstatus != BUILDSUBTREE_FAILURE))
		{
			status = BUILDSUBTREE_SAMEPROCESS;
				if(subtreeRoot && (subtreeRoot->desc).owner != (intNode->desc).owner)
				{
					MsgUpdateMinMax msgDoneTree;
					msgDoneTree.parent = (subtreeRoot->desc).owner;
					msgDoneTree.pRoot = reinterpret_cast<long int>(intNode);
					msgDoneTree.pLeaf = reinterpret_cast<long int>(subtreeRoot->parent);
					msgDoneTree.isLeft = isLeft;
					GAL_SendMessage((subtreeRoot->desc).owner,MESSAGE_DONESUBTREE,&msgDoneTree);
					status = BUILDSUBTREE_FAILURE;
				}
	
		}
		return status;

	}
	
}

void GAL::GAL_PrintGraph()
{
#ifdef TRAVERSAL_PROFILE
	//TODO:uncomment the below line when collecting distributed tree profile.
	if(procRank == 0)
	{
		std::vector<long int>::iterator iter = pipelineStage.begin();
		for(;iter!=pipelineStage.end();iter++)
		{
			GAL_Vertex* pRoot = reinterpret_cast<GAL_Vertex*>(*iter);
			/*if((pRoot->level != 0) && (pRoot->leftChild || pRoot->rightChild))
			{
				long int count=0;
				GALHelper_CountSubtreeNodes(pRoot, count);
				printf("Subtree %ld Points_Visited %ld\n",*iter,count);
			}*/
			if(pRoot->level == 0)
			{
				long int count=0;
				GALHelper_CountSubtreeNodes(pRoot, count);
				printf("Subtree %ld Points_Visited %ld\n",*iter,count);
				double bof=0.;
				GALHelper_GetBlockOccupancyFactor(pRoot, bof);
				printf("Bof:  %f\n",bof/count);
			}
		}
	}
	

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

#ifdef DYNAMIC_LB2
	int totalRepSubtrees=0;
	reduce(communicator(pg),numReplicas, totalRepSubtrees, std::plus<int>(),0);
	if(numReplicas> 0)
		printf("%d: number of replicated subtrees:%d\n",procRank,numReplicas);
	if(procRank == 0)
		printf("Total number of replicated subtrees:%d\n",totalRepSubtrees);
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

		if(tmpChildNode->pseudoLeaf && tmpChildNode->isLeftChild)
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
				ret = GAL_TraverseHelper(vis,rootNode, NULL);
				
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
							msgTerminate.pLeft = ret;
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
						if(pRoot->leftChild || pRoot->rightChild)
							((GAL_PseudoRoot*)pRoot)->parents2[procId]=msgTraverse.pLeaf;
#endif
#ifdef DYNAMIC_LB2
						((GAL_PseudoRoot*)pRoot)->numPointsSeen+=msgTraverse.l.size();
						if(!loadBalanced && (numReplicas<STATIC_LB_TRIGGER)&& ((GAL_PseudoRoot*)pRoot)->numPointsSeen >= DYNAMIC_LB_TRIGGER)
						//if(!loadBalanced && ((procRank==2)||(procRank==4)||(procRank==10)||(procRank==3)||(procRank==5)||(procRank==12)||(procRank==6)||(procRank==8)||(procRank==13)||(procRank==14)||(procRank==1)) && ((GAL_PseudoRoot*)pRoot)->numPointsSeen >= DYNAMIC_LB_TRIGGER)
						//if(!loadBalanced && ((procRank==2)||(procRank==4)||(procRank==10)) && ((pRoot->label==14403) || (pRoot->label==16764)||(pRoot->label==17551)||(pRoot->label==13616)))
						{
							if((pRoot->leftChild!=NULL) || (pRoot->rightChild!=NULL))
							{
								bool requestAlreadySent=false;
								std::pair<std::map<GAL_Vertex*,int>::iterator,bool> requestPending;
								requestPending = replicatedSubtreeRequests.insert(std::make_pair(pRoot,procId));
								if(!(requestPending.second))
								{
									requestAlreadySent=true;
								}

								MsgReplicateSubtree msgReplicateSubtree;
								msgReplicateSubtree.blkStart=msgTraverse.blkStart;
								if(msgTraverse.pSibling==0)	
									msgReplicateSubtree.l=msgTraverse.l;

								long int parent = reinterpret_cast<long int>(pRoot->parent);
								/*std::vector<RepPLeaf>::iterator pLeafIter=(((GAL_PseudoRoot*)(pRoot))->parents).begin();
								for(;pLeafIter!= (((GAL_PseudoRoot*)(pRoot))->parents).end();pLeafIter++)
								{
									if((pLeafIter->first) == procId)
									{
										parent = pLeafIter->second;
										break;
									}
								}*/
								if(procId != 0)
									parent = (((GAL_PseudoRoot*)(pRoot))->parents2[procId];
								msgReplicateSubtree.pLeaf=parent;
									
								//get subtree info into a string and broadcast to all for replication.
								int childNum=pRoot->isLeftChild?0:1;
								if(requestAlreadySent)
								{
									msgReplicateSubtree.isLeftChild=pRoot->isLeftChild;
									GAL_SendMessage(procId,MESSAGE_REPLICATE_SUBTREE, &msgReplicateSubtree);
									break;
								}
								numReplicas++;
								/*if(childNum==0)
									printf("%d: requesting %d to create a replicated left subtree at %p\n",procRank,procId,parent);
								else
									printf("%d: requesting %d to create a replicated right subtree at %p\n",procRank,procId,parent);*/
								/*std::vector<std::string> tmpString;
								std::pair<std::map<GAL_Vertex*,std::vector<std::string> >::iterator,bool> cacheStatus;
								cacheStatus=subtreeCache.insert(std::make_pair(pRoot,tmpString));
								if(!(cacheStatus.second))
								{
									msgReplicateSubtree.data=(cacheStatus.first)->second;	
								}
								else*/
								{
									GALHelper_SaveSubtreeAsString(pRoot, msgReplicateSubtree.data, -1, childNum, requestAlreadySent);
									//(cacheStatus.first)->second=msgReplicateSubtree.data;
								}
								//GAL_SendMessage(procId,MESSAGE_REPLICATE_SUBTREE, &msgReplicateSubtree);
								for(int i=0;i<numProcs;i++)
								{
									if(i!=procRank)
									{
										assert(pRoot->parentDesc == 0);
										long int parent = reinterpret_cast<long int>(pRoot->parent);
										/*std::vector<RepPLeaf>::iterator pLeafIter=(((GAL_PseudoRoot*)(pRoot))->parents).begin();
										for(;pLeafIter!= (((GAL_PseudoRoot*)(pRoot))->parents).end();pLeafIter++)
										{
											if((pLeafIter->first) == i)
											{
												parent = pLeafIter->second;
												break;
											}
										}*/
										if(i!=0)
											parent = (((GAL_PseudoRoot*)(pRoot))->parents2[i];
										msgReplicateSubtree.pLeaf=parent;
										GAL_SendMessage(i,MESSAGE_REPLICATE_SUBTREE, &msgReplicateSubtree);
									}
								}
								break;
							}
						}	
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
#ifdef DYNAMIC_LB
				case MESSAGE_REPLICATE_SUBTREE:
				{
					MsgReplicateSubtree msgReplicateSubtree;
					receive_oob(pg, msg.first, msg.second, msgReplicateSubtree);
					GAL_Vertex* pRoot=NULL, *pLeaf=NULL;
					pLeaf=reinterpret_cast<GAL_Vertex*>(msgReplicateSubtree.pLeaf);
#ifdef PERFCOUNTERS
						compTimer.Start();
#endif
					if(msgReplicateSubtree.data.size()!=0)
					{
						GALHelper_ReplicateSubtreeFromString(msgReplicateSubtree,&pRoot, pLeaf,msg.first);
						pRoot->parent = pLeaf;
					}
					else
					{
						pRoot=msgReplicateSubtree.isLeftChild?pLeaf->leftChild:pLeaf->rightChild;
					}
					assert(pRoot!=NULL);
					if(msgReplicateSubtree.blkStart.first==procRank)
					{
						vis->GALVisitor_SetLocalData(msgReplicateSubtree.l, msgReplicateSubtree.blkStart);
						MsgTraverse msgTraverse;
						vis->GALVisitor_GetLocalData(msgTraverse.l);
						msgTraverse.blkStart = msgReplicateSubtree.blkStart;
						msgTraverse.pLeaf = reinterpret_cast<long int>(pLeaf);
						if(pRoot->isLeftChild)
						{
							msgTraverse.pSibling = reinterpret_cast<long int>(pRoot->parent->rightChild);
							msgTraverse.siblingDesc = pRoot->parent->rightDesc;
						}
						else
							msgTraverse.pSibling = 0;
						int status = GALHelper_DeAggregateBlockAndSend(vis, &pRoot, msgTraverse, true);
						if(status==STATUS_TRAVERSE_COMPLETE)
						{
							long int parent = reinterpret_cast<long int>(pRoot->parent);
							/*std::vector<RepPLeaf>::iterator pLeafIter=(((GAL_PseudoRoot*)(pRoot))->parents).begin();
							for(;pLeafIter!= (((GAL_PseudoRoot*)(pRoot))->parents).end();pLeafIter++)
							{
								if((pLeafIter->first) == procRank)
								{
									parent = pLeafIter->second;
									break;
								}
							}*/
							if((pRoot->pseudoRoot) && (procRank != 0))
								parent = (((GAL_PseudoRoot*)(pRoot))->parents2[procRank];
							status = GALHelper_HandleMessageTraverseBackward(pLeaf, vis, msgReplicateSubtree.blkStart);
							if(status == 1)
							{
								vis->GALVisitor_RemoveBlockStack();		
							}
						}
					}
#ifdef PERFCOUNTERS
						if(compTimer.IsRunning())
							compTimer.Stop();
#endif

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
		GAL_Vertex* pLeaf = reinterpret_cast<GAL_Vertex*>(pLeafPointsPerProcess[msgIter->label].pLeaf); 
		if((pLeaf->leftChild == NULL) && msgIter->pLeftChild)
		{
			pLeaf->leftDesc = msgIter->leftDesc;	
			pLeaf->leftChild = reinterpret_cast<GAL_Vertex*>(msgIter->pLeftChild);
#ifndef HYBRID_BUILD
			if(pLeaf->leftDesc == procRank)
			{
				GAL_PseudoRoot* child = reinterpret_cast<GAL_PseudoRoot *>(msgIter->pLeftChild);
				//(child->parents).push_back(std::make_pair(procRank, pLeafPointsPerProcess[msgIter->label].pLeaf));
				child->parents2[procRank]=pLeafPointsPerProcess[msgIter->label].pLeaf;
			}
#endif
		}

		if((pLeaf->rightChild == NULL) && (msgIter->pRightChild))
		{
			pLeaf->rightDesc = msgIter->rightDesc;	
			pLeaf->rightChild = reinterpret_cast<GAL_Vertex*>(msgIter->pRightChild);	
#ifndef HYBRID_BUILD
			if(pLeaf->rightDesc == procRank)
			{
				GAL_PseudoRoot* child = reinterpret_cast<GAL_PseudoRoot *>(msgIter->pRightChild);
				//(child->parents).push_back(std::make_pair(procRank, pLeafPointsPerProcess[msgIter->label].pLeaf));
				child->parents2[procRank]=pLeafPointsPerProcess[msgIter->label].pLeaf;
			}
#endif
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

	std::pair<std::map<long int, int>::iterator, bool> ret = pLeafMap.insert(std::make_pair(pLeaf,0));
	if(ret.second == true)
		(ret.first)->second = nextprocessid;

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
		count += 1;//ver->pointsVisited;
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
			{
				GAL_Vertex* tmp=ver->leftChild;
				/*if(tmp->level == 11)
					return;*/
				GALHelper_CountSubtreeNodes(tmp,count);
			}
		}
		if(ver->rightChild)
		{
			if(ver->rightDesc == procRank)
			{
				GAL_Vertex* tmp=ver->rightChild;
				/*if(tmp->level == 11)
					return;*/
				GALHelper_CountSubtreeNodes(tmp,count);
			}
		}
			
		
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
		output<<msgIter->label<<" "<<msgIter->pLeftChild<<" "<<(msgIter->leftDesc)<<" "<<msgIter->pRightChild<<" "<<(msgIter->rightDesc);
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
			int label,leftDesc, rightDesc;
			long int leftChild,rightChild;
			input >> label;
			input >> leftChild;
			input >> leftDesc;
			input >> rightChild;
			input >> rightDesc;
			pLeaf.label=label;
			pLeaf.pLeftChild=leftChild;
			pLeaf.pRightChild=rightChild;
			pLeaf.leftDesc=leftDesc;
			pLeaf.rightDesc=rightDesc;
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
	int isLeaf=((node->leftChild==NULL) && (node->rightChild==NULL))?1:0;
	if(node->leftChild && (node->leftDesc!=procRank))
		pLeaf=1;
	if(node->rightChild && (node->rightDesc!=procRank))
		pLeaf=1;
		
	fp<<node->label<<" "<<parentId<<" "<<childNum<<" "<<(short int)(node->level)<<" "<<pLeaf<<" "<<isLeaf;
	if(pLeaf==1)
	{
		fp<<" "<<reinterpret_cast<long int>(node->leftChild)<<" "<<(node->leftDesc)<<" "<<reinterpret_cast<long int>(node->rightChild)<<" "<<(node->rightDesc);
	}
	if(isLeaf)
	{
		fp<<" "<<(float)(node->cp.cofm.x);
		fp<<" "<<(float)(node->cp.cofm.y);
		fp<<" "<<(float)(node->cp.cofm.z);
	}
	fp<<std::endl;
	if(isLeaf)
	{
		return;
	}
	
	if(node->leftChild && (node->leftDesc == procRank))
	{
		GALHelper_SaveSubtreeInFile(node->leftChild,node->label,0,fp);
	}
	if(node->rightChild && (node->rightDesc == procRank))
	{
		GALHelper_SaveSubtreeInFile(node->rightChild,node->label,1,fp);
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
				}
				else
					node = GAL_CreateVertex(VTYPE_NORMAL);

				parentNode = reinterpret_cast<GAL_Vertex*>(repVertexTable[parentId]);
				node->parentDesc = procRank;
			}
			
			node->label=label;
			node->level=level;
			node->parent = parentNode;
			if(childNum==0)
			{
				if(recordNum != 1)
				{
					parentNode->leftChild=node;
					parentNode->leftDesc = procRank;
				}
			
				node->isLeftChild=true;
			}
			else
			{
				if(recordNum != 1)
				{
					parentNode->rightChild=node;
					parentNode->rightDesc = procRank;
				}
				node->isLeftChild=false;
			}

			long int vertexPtr = reinterpret_cast<long int>(node);
			repVertexTable.insert(std::make_pair(label,vertexPtr));	

			if(isLeaf)
			{
				input >> (node->cp).cofm.x;
				input >> (node->cp).cofm.y;
				input >> (node->cp).cofm.z;

			}
			else
			{
				if(isPLeaf == 1)
				{
					long int leftChild, rightChild;
					int leftDesc, rightDesc;
					input>>leftChild;
					input>>leftDesc;
					input>>rightChild;
					input>>rightDesc;
					/*node->leftChild=reinterpret_cast<GAL_Vertex*>(leftChild);
					node->rightChild=reinterpret_cast<GAL_Vertex*>(rightChild);
					node->leftDesc=leftDesc;
					node->rightDesc=leftDesc;*/
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
	int isLeaf=((node->leftChild==NULL) && (node->rightChild==NULL))?1:0;
	if(node->leftChild && (node->leftDesc!=procRank))
		pLeaf=1;
	if(node->rightChild && (node->rightDesc!=procRank))
		pLeaf=1;
		
	stroutput<<node->label<<" "<<parentId<<" "<<childNum<<" "<<(short int)(node->level)<<" "<<pLeaf<<" "<<isLeaf;
	if(pLeaf==1)
	{
		stroutput<<" "<<reinterpret_cast<long int>(node->leftChild)<<" "<<(node->leftDesc)<<" "<<reinterpret_cast<long int>(node->rightChild)<<" "<<(node->rightDesc);
	}
	if(isLeaf)
	{
		stroutput<<" "<<(float)(node->cp.cofm.x);
		stroutput<<" "<<(float)(node->cp.cofm.y);
		stroutput<<" "<<(float)(node->cp.cofm.z);
	}
	subtreeStr.push_back(stroutput.str());

	if(isLeaf || requestPending)
	{
		return;
	}
	
	if(node->leftChild && (node->leftDesc == procRank))
	{
		GALHelper_SaveSubtreeAsString(node->leftChild,subtreeStr,node->label,0, false);
	}
	if(node->rightChild && (node->rightDesc == procRank))
	{
		GALHelper_SaveSubtreeAsString(node->rightChild,subtreeStr,node->label,1, false);
	}
	
	return;
}

void GAL::GALHelper_CreateSubtreeFromString(MsgReplicateSubtree& msgCloneSubtree, GAL_Vertex** pRoot, GAL_Vertex* pLeaf, int pLeafOwner)
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
			}
			else
				node = GAL_CreateVertex(VTYPE_NORMAL);

			parentNode = reinterpret_cast<GAL_Vertex*>(repVertexTable[parentId]);
			node->parentDesc = procRank;
		}
		
		node->label=label;
		node->level=level;
		node->parent = parentNode;
		if(childNum==0)
		{
			if(recordNum != 1)
			{
				parentNode->leftChild=node;
				parentNode->leftDesc = procRank;
			}
		
			node->isLeftChild=true;
		}
		else
		{
			if(recordNum != 1)
			{
				parentNode->rightChild=node;
				parentNode->rightDesc = procRank;
			}
			node->isLeftChild=false;
		}

		long int vertexPtr = reinterpret_cast<long int>(node);
		repVertexTable.insert(std::make_pair(label,vertexPtr));	

		if(isLeaf)
		{
			strinput >> (node->cp).cofm.x;
			strinput >> (node->cp).cofm.y;
			strinput >> (node->cp).cofm.z;

		}
		else
		{
			if(isPLeaf == 1)
			{
				long int leftChild, rightChild;
				int leftDesc, rightDesc;
				strinput>>leftChild;
				strinput>>leftDesc;
				strinput>>rightChild;
				strinput>>rightDesc;
				/*node->leftChild=reinterpret_cast<GAL_Vertex*>(leftChild);
				node->rightChild=reinterpret_cast<GAL_Vertex*>(rightChild);
				node->leftDesc=leftDesc;
				node->rightDesc=leftDesc;*/
			}

		}

	}
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
			std::pair<std::map<GAL_Vertex*, int>::iterator, bool> alreadyReplicated;
			alreadyReplicated = replicatedSubtreeTable.insert(std::make_pair(pLeaf,(1<<childNum)));
			if(!(alreadyReplicated.second))
			{

				int replicatedChild = ((alreadyReplicated.first)->second & (1<<childNum));
				if(replicatedChild)
				{
					if(!childNum)
					{
						*pRoot=pLeaf->leftChild;
						//printf("%d: already created replicated left subtree at %p\n",procRank,pLeaf);
					}
					else
					{
						*pRoot=pLeaf->rightChild;	
						//printf("%d: already created replicated right subtree at %p\n",procRank,pLeaf);
					}
					return;
				}
				((alreadyReplicated.first)->second |= (1<<childNum));
			}
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
		

		node->label=label;
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
			strinput >> (node->cp).cofm.x;
			strinput >> (node->cp).cofm.y;
			strinput >> (node->cp).cofm.z;
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


#ifdef DYNAMIC_LB
void GAL::GALHelper_ReplicateSubtrees()
{
	PBGL_oobMessage msg;
	bool done=false;
	bneckList=(int**)malloc(sizeof(int*)*numProcs);
	for(int i=0;i<numProcs;i++)
		bneckList[i]=(int *)malloc(sizeof(int)*NUMB);
	for(int i=0;i<numProcs;i++)
	{
		if((i==4)||(i==10))
		{
			for(int j=0;j<NUMB;j++)
			{
					switch(j)
					{
					case 0:	bneckList[i][j]=67046;
						break;
					case 1: bneckList[i][j]=51326;
						break;
					case 2: bneckList[i][j]=52113;
						break;
					case 3: bneckList[i][j]=54470;
						break;
					case 4: bneckList[i][j]=55257;
						break;
					case 5: bneckList[i][j]=67833;
						break;
					case 6: bneckList[i][j]=63902;
						break;
					}
			}
		}
		else if(i==12)
		{
			for(int j=0;j<NUMB;j++)
			{
					switch(j)
					{
					case 0:	bneckList[i][j]=1024;
						break;
					case 1: bneckList[i][j]=3379;
						break;
					case 2: bneckList[i][j]=1809;
						break;
					case 3: bneckList[i][j]=12014;
						break;
					case 4: bneckList[i][j]=4164;
						break;
					case 5: bneckList[i][j]=-1;
						break;
					case 6: bneckList[i][j]=-1;
						break;
					}
			}
		}
		else if(i==1)
		{
			for(int j=0;j<NUMB;j++)
			{
					switch(j)
					{
					case 0:	bneckList[i][j]=1024;
						break;
					case 1: bneckList[i][j]=5734;
						break;
					case 2: bneckList[i][j]=2594;
						break;
					case 3: bneckList[i][j]=8874;
						break;
					case 4: bneckList[i][j]=7304;
						break;
					case 5: bneckList[i][j]=107784;
						break;
					case 6: bneckList[i][j]=109354;
						break;
					default:bneckList[i][j]=-1;
						break;
					}
			}
		}
		else if(i==2)
		{
			for(int j=0;j<NUMB;j++)
			{
					switch(j)
					{
					case 0:	bneckList[i][j]=-1;
						break;
					case 1: bneckList[i][j]=-1;
						break;
					case 2: bneckList[i][j]=67919;
						break;
					case 3: bneckList[i][j]=52179;
						break;
					case 4: bneckList[i][j]=52179;
						break;
					case 5: bneckList[i][j]=54540;
						break;
					case 6: bneckList[i][j]=55327;
						break;
					default:bneckList[i][j]=-1;
						break;
					}
			}
		}
		else
			for(int j=0;j<NUMB;j++)
				bneckList[i][j]=-1;	
	}
	if(STATIC_LB_TRIGGER >= pipelineStage.size())
		STATIC_LB_TRIGGER=pipelineStage.size()-1;
	//printf("%d entering\n",procRank);
	while (!done)
	{
		PBGL_AsyncMsg pollMsg = pg.poll();
		if(!pollMsg)
		{
				int ret=STATUS_FAILURE;
				if(procRank!=0)
					ret = GALHelper_GetRandomSubtreeAndBroadcast();
				if(ret==STATUS_FAILURE)
				{
					if(!readyToExit)
					{
						readyToExit = true;
						if(readyToExitList.size() != numProcs)
						{
							readyToExitList.insert(procRank);
							//printf("%d broadcasting exit\n",procRank);
							for(int i=0;i<numProcs;i++)
							{
								if(i!=procRank)
								{
									GAL_SendMessage(i,MESSAGE_READYTOEXIT,&procRank); 
								}
							}
							if(readyToExitList.size() == numProcs)
								done = true;
						}
					}
				}
		}
		else
		{
			msg = pollMsg.get();
			switch(msg.second)
			{
				case MESSAGE_REPLICATE_SUBTREE:
				{
					MsgReplicateSubtree msgReplicateSubtree;
					receive_oob(pg, msg.first, msg.second, msgReplicateSubtree);
					//printf("%d receiving replication from %d\n",procRank,msg.first);
					GAL_Vertex* pRoot=NULL, *pLeaf=NULL;
					pLeaf=reinterpret_cast<GAL_Vertex*>(msgReplicateSubtree.pLeaf);
					//printf("%d replicated subtree at %d(from %d)\n",procRank,pLeaf->label,msg.first);
					GALHelper_ReplicateSubtreeFromString(msgReplicateSubtree,&pRoot, pLeaf,msg.first);
					assert(pRoot!=NULL);
					pRoot->parent = pLeaf;
				}
				break;
				case MESSAGE_READYTOEXIT:
				{
					int doneProcId;
					receive_oob(pg, msg.first, msg.second, doneProcId);
					//printf("%d receiving exit from %d\n",procRank,msg.first);
					readyToExitList.insert(doneProcId);
					if(readyToExitList.size() == numProcs)
						done = true;
				}
				break;
				default: break;
			}
		}
	}
	//printf("%d exiting\n",procRank);
	//synchronize(pg);
	readyToExit=false;
	readyToExitList.clear();
}

int GAL::GALHelper_GetRandomSubtreeAndBroadcast()
{
	MsgReplicateSubtree subtree;
	int i=0,size=pipelineStage.size();
	//get random number within range 0 and size;
	long index = GALHelper_GetRandom(size);
	if(index == 0)
		index++;
	GAL_Vertex* pRoot=pipelineStage[index];
	assert(index <= size);
	if(numReplicas>=STATIC_LB_TRIGGER)
		return STATUS_FAILURE;
	
	if(pRoot==NULL)
	{
		return STATUS_FAILURE;
	}
	
	if(pRoot->parent==NULL)
	{
		return STATUS_FAILURE;
	}

	std::pair<std::map<GAL_Vertex*,int>::iterator,bool> requestPending;
	requestPending = replicatedSubtreeRequests.insert(std::make_pair(pRoot,procRank));
	if(!(requestPending.second))
	{
		return STATUS_SUCCESS;
	}

	numReplicas++;
	int childNum=pRoot->isLeftChild?0:1;
	GALHelper_SaveSubtreeAsString(pRoot, subtree.data, -1, childNum, false);
	for(int i=0;i<numProcs;i++)
	{
		if(i!=procRank)
		{
			assert(pRoot->parentDesc == 0);
			long int parent = reinterpret_cast<long int>(pRoot->parent);
			//std::vector<RepPLeaf>::iterator pLeafIter=(((GAL_PseudoRoot*)(pRoot))->parents).begin();
			std::map<int, long int>::iterator pLeafIter=(((GAL_PseudoRoot*)(pRoot))->parents2).begin();
			//for(;pLeafIter!= (((GAL_PseudoRoot*)(pRoot))->parents).end();pLeafIter++)
			for(;pLeafIter!= (((GAL_PseudoRoot*)(pRoot))->parents2).end();pLeafIter++)
			{
				if((pLeafIter->first) == i)
				{
					parent = pLeafIter->second;
					break;
				}
			}
			subtree.pLeaf=parent;
			GAL_SendMessage(i,MESSAGE_REPLICATE_SUBTREE, &subtree);
		}
	}
	return STATUS_SUCCESS;
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

bool GAL::GALHelper_IsBottleneckSubtree(int label, int** bneckList)
{
	for(int i=0;i<numProcs;i++)
	{
		for(int j=0;j<NUMB;j++)
		{
			if(bneckList[i][j]==label)
			{
				bneckList[i][j]=-1;
				return true;
			}
		}
	}
	return false;
}
#endif

void GAL::print_treetofile(FILE* fp)
{
	print_preorder(rootNode, fp);
}

void GAL::print_preorder(GAL_Vertex* node, FILE* fp)
{
	fprintf(fp,"%d ",node->label);
	fprintf(fp,"%f %f %f %f",node->cp.cofm.x,node->cp.cofm.y, node->cp.cofm.z, node->cp.ropen);
	fprintf(fp,"\n");
	if(node->leftChild)
		print_preorder(node->leftChild,fp);

	if(node->rightChild)
		print_preorder(node->rightChild,fp);
}

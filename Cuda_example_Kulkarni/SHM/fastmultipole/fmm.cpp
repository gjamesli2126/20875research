/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "fmm.h"
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<assert.h>
#include<math.h>
#include<pthread.h>
#include<string>
#include<set>
#ifdef PAPI
#include"papi.h"
void handle_error(int retval)
{
	printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	exit(1);
}
#endif

#ifdef METRICS
int numberOfVertices=0;
int numberOfTraversals=0;
int numberOfNodesVisited=0;
std::vector<Vertex*> subtrees;
#endif
double treeConstnTime;
long int labelCount;
int maxdepth=0;
std::vector<Vertex*> allLeaves;
Vertex* rootNode;

long int get_elapsed_usec(struct timeval& start_time, struct timeval& end_time) 
{
	int sec = end_time.tv_sec - start_time.tv_sec;
	int usec = end_time.tv_usec - start_time.tv_usec;
	return (sec * 1000000) + usec;
}

int numThreads=1;
pthread_mutex_t fmm_mutex;
int lock(pthread_mutex_t* h_mutex) { pthread_mutex_lock(h_mutex); }
int unlock(pthread_mutex_t* h_mutex) { pthread_mutex_unlock(h_mutex); }

static void *thread_entry(void * arg) 
{
	struct targs * t = (struct targs*)arg;
	t->func(t->funcIn,t->start, t->end, t->funcOut);
}

int *distribute_among(int initStart, int end, int numt) 
{
	int *ret = new int[numt * 2];
	int start = initStart;
	int i;
	int num;
	for (i = 0; i < numt; i++) {
		num = (end - start) / (numt - i);
		ret[2 * i] = start;
		ret[2 * i + 1] = start + num;
		start += num;
	}
	return ret;
}

Vertex* CreateVertex()
{
#ifdef METRICS
	numberOfVertices++;
#endif
	labelCount++;
	Vertex* node; 
	node = new Vertex;
	node->label = labelCount;
	return node;
}

int ConstructQuadTreeAndTraverse(Point* points, Box& boundingBox, int numPoints, bool hasMoreData)
{
	struct timeval start_time, end_time;
	FuncIn inParam;
	inParam.allLeaves = &allLeaves;
	inParam.start = 0;
	inParam.end = allLeaves.size();
	double sx= FLT_MAX, sy = FLT_MAX;
	double ex=-FLT_MAX, ey = -FLT_MAX;
	gettimeofday(&start_time, NULL);
	if(!rootNode)
	{
		rootNode = CreateVertex();
		rootNode->vData = new VertexData(0.0);
	}
	rootNode->box = boundingBox;
	rootNode->level=0;
	Vec center(0.0);
	double dia;
	ComputeBoundingBoxParams(boundingBox, center, dia);
	for(int i=0;i<numPoints;i++)
	{
		rootNode->vData->p->mass += points[i].mass;
		BuildSubTree_Quad(rootNode, &points[i], center, dia, 1, 1, false);
	}

	for(int i=0;i<MAX_CHILDREN;i++)	
		if(rootNode->pChild[i])
			UpdateNeighbors(rootNode->pChild[i]);

	GetLeafNodes(rootNode);
	gettimeofday(&end_time, NULL);
	treeConstnTime += get_elapsed_usec(start_time, end_time)/(double)CLOCKS_PER_SEC;
	if(!hasMoreData)
	{
	int * outParam = new int[numThreads];
	long int consumedTime_TD, consumedTime_BU; 
	printf("Tree construction time %f\n",treeConstnTime);
	printf("maximum depth : %d\n",maxdepth);
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
	retval = PAPI_add_event(EventSet, PAPI_L2_TCM);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// Total L3 cache accesses = total memory accesses. Needed for computing L3 miss rate. On Wabash, there are 3 layers of cache. 
	retval = PAPI_add_event(EventSet, PAPI_L2_TCA);
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
	printf("Number of leaves %d\n",allLeaves.size());
	/*for(int i=0;i<allLeaves.size();i++)
	{
		printf("%d: ",allLeaves[i]->level);
		int numPerCell = allLeaves[i]->myPoints.size();
		for(int j=0;j<numPerCell;j++)
		{
			printf("%d ",allLeaves[i]->myPoints[j]->id);
		}
		printf("\n");
	}*/
	//Step 2
	gettimeofday(&start_time, NULL);
#ifdef PAPI
	retval = PAPI_start(EventSet);
	if (retval != PAPI_OK) handle_error(retval);
#endif
	/*TraverseDown_Recursive(rootNode,2);
	TraverseDown_Recursive(rootNode,3);*/
	TraverseDownFused_Recursive(rootNode);

	gettimeofday(&end_time, NULL);
	consumedTime_TD = get_elapsed_usec(start_time, end_time);
	
	//Steps 3 and 4
	gettimeofday(&start_time, NULL);
	/*inParam.stepNum = 3;
	parallel_for(TraverseUp, 0, allLeaves.size(),&inParam, outParam);*/
	inParam.stepNum = 4;
	parallel_for(TraverseUp, 0, allLeaves.size(), &inParam, outParam);
	gettimeofday(&end_time, NULL);
	consumedTime_BU = get_elapsed_usec(start_time, end_time);
	/*for(int i=0;i<numPoints;i++)
	{
		printf("%d (%f %f) %f\n",points[i].id,points[i].coordX, points[i].coordY, points[i].potential);	
	}*/
#ifdef PAPI
	/* Stop the counters */
	retval = PAPI_stop(EventSet, values);
	if (retval != PAPI_OK) 
		handle_error(retval);
	double missRate = values[0]/(double)(values[1]);
	double  CPI = values[2]/(double)(values[3]);
	printf("L3 Miss Rate:(%ld / %ld) = %f CPI (Total Cycles/ Total Instns): (%ld/%ld) = %f\n",values[0],values[1],missRate,values[2],values[3],CPI);
#endif
	delete [] outParam;
	printf("Top-down traversal time %f\n",consumedTime_TD/(double)CLOCKS_PER_SEC);
	printf("Bottom up traversal time %f\n",consumedTime_BU/(double)CLOCKS_PER_SEC);
	
		
	}
	return 0;
}


int BuildSubTree_Quad(Vertex* subtreeRoot, Point* point, Vec& center, double dia, int depth, int DOR, bool clonePoint)
{
	int ret = 0;
	assert(subtreeRoot != NULL);
	
	int space=0;
	Vec offset(0.0f);

	if (center.pt[0] < point->coordX) 
	{
		space += 1;
		offset.pt[0] = dia/2;
	}
	if (center.pt[1] < point->coordY) 
	{
		space += 2;
		offset.pt[1] = dia/2;
	}

	Vertex *child = subtreeRoot->pChild[space];
	if (child == NULL)
	{
		Vertex *n=NULL;
		n = CreateVertex();
		VertexData* newvData = new VertexData(point,clonePoint);
		n->vData  =  new VertexData(point->mass);
		n->vData->next = newvData; 
		n->isLeaf = true;
		n->level = depth;
		n->numPointsInCell++;
		subtreeRoot->pChild[space]=n;
		n->parent = subtreeRoot;
		switch(space)
		{
			case 2:
				n->box.startX = subtreeRoot->box.startX;
				n->box.startY = subtreeRoot->box.startY+dia/2;
				break;
			case 3:
				n->box.startX = subtreeRoot->box.startX+dia/2;
				n->box.startY = subtreeRoot->box.startY+dia/2;
				break;
			case 0: 
				n->box.startX = subtreeRoot->box.startX;
				n->box.startY = subtreeRoot->box.startY;
				break;
			case 1: 
				n->box.startX = subtreeRoot->box.startX+dia/2;
				n->box.startY = subtreeRoot->box.startY;
				break;
			default:assert(0);
				break;
		}
		n->box.endX = n->box.startX + dia/2;
		n->box.endY = n->box.startY + dia/2;
		if(depth > maxdepth)
			maxdepth = depth;
	}
	else 
	{
		child->vData->p->mass += point->mass;
		if (child->isLeaf)
		{
			VertexData* vData = child->vData->next;
			if((vData->p->coordX == point->coordX) && (vData->p->coordY == point->coordY))
			{
				printf("DUPLICATE\n");
				//vData->p->mass += point->mass;
				//return 0;
			}
			
			if((child->numPointsInCell == NUM_POINTS_PER_CELL) && (DOR != MAX_LEVELS))
			{
				child->isLeaf = false;
				child->numPointsInCell=0;
				double halfR = dia/2;
				/*if(halfR != (child->box.endX-child->box.startX))
					printf("Debug break\n");
				if(halfR != (child->box.endY-child->box.startY))
					printf("Debug break\n");*/
				//TODO
				/*assert(halfR == (child->box.endX-child->box.startX));
				assert(halfR == (child->box.endY-child->box.startY));*/
				Vec childBoxCenter(0);
				childBoxCenter.pt[0] = child->box.startX + halfR/2;
				childBoxCenter.pt[1] = child->box.startY + halfR/2;
				VertexData* otherVData = vData;
				while(otherVData)
				{
					VertexData* nextVData = otherVData->next;
					Point* cPoint = otherVData->p;
					int childNum = ComputeChildNumber(childBoxCenter, cPoint);
					if(child->pChild[childNum] == NULL)
					{
						otherVData->next=NULL;
						Vertex* newNode;
						newNode = CreateVertex();
						newNode->isLeaf = true;
						newNode->level = child->level + 1;
						newNode->vData  =  new VertexData(cPoint->mass);
						newNode->vData->next = otherVData;
						newNode->numPointsInCell++;
						child->pChild[childNum] = newNode;
						newNode->parent = child;
						switch(childNum)
						{
							case 2:
								newNode->box.startX = child->box.startX;
								newNode->box.startY = child->box.startY+halfR/2;
								break;
							case 3:
								newNode->box.startX = child->box.startX+halfR/2;
								newNode->box.startY = child->box.startY+halfR/2;
								break;
							case 0: 
								newNode->box.startX = child->box.startX;
								newNode->box.startY = child->box.startY;
								break;
							case 1: 
								newNode->box.startX = child->box.startX+halfR/2;
								newNode->box.startY = child->box.startY;
								break;
							default:assert(0);
								break;
						}
						newNode->box.endX = newNode->box.startX+halfR/2;
						newNode->box.endY = newNode->box.startY+halfR/2;
						if(child->level+1 > maxdepth)
							maxdepth = child->level+1;
					}
					else
					{
						otherVData->p=NULL; //to avoid deleting the containing point, which will be inserted into a new cell (below).
						delete otherVData;
						ret = BuildSubTree_Quad(child,cPoint,childBoxCenter, halfR,depth+1, DOR+1, false);
					}
					otherVData = nextVData;
				}
				child->vData->next = NULL;
				ret = BuildSubTree_Quad(child,point,childBoxCenter, halfR,depth+1, DOR+1, clonePoint);
			}
			else
			{
				VertexData* newvData = new VertexData(point, clonePoint);
				newvData->next = child->vData->next;
				child->vData->next = newvData;
				child->numPointsInCell++;
			}
		}
		else
		{
			double halfR = dia/2;
			Vec newCenter(0.0);
			for(int i=0;i<DIMENSION;i++)
				newCenter.pt[i] = (center.pt[i]-halfR/2)+offset.pt[i];
			ret = BuildSubTree_Quad(child,point,newCenter, halfR,depth+1, DOR+1, clonePoint);
		}
	}

	return ret;
}

void UpdateNeighbors(Vertex* node)
{
	Vertex* parent = node->parent;
	for(int i=0;i<parent->neighbors.size();i++)	
	{
		Vertex* parNeighbor = parent->neighbors[i];
		for(int j=0;j<MAX_CHILDREN;j++)
			if(parNeighbor->pChild[j] && AreAdjacent(parNeighbor->pChild[j]->box, node->box))
				node->neighbors.push_back(parNeighbor->pChild[j]);
	}
	
	for(int i=0;i<MAX_CHILDREN;i++)	
		if(parent->pChild[i] && (parent->pChild[i] != node))
			node->neighbors.push_back(parent->pChild[i]);

	for(int i=0;i<MAX_CHILDREN;i++)	
		if(node->pChild[i])
			UpdateNeighbors(node->pChild[i]);
}

void TraverseUp(void* inParam, int start, int end, void* outParam)
{
	//int start = ((FuncIn*)inParam)->start;
	//int end = ((FuncIn*)inParam)->end;
	int step = ((FuncIn*)inParam)->stepNum;
	std::vector<Vertex*>* allLeaves=((FuncIn*)inParam)->allLeaves; 
	//printf("thread start %d end %d\n", start, end);
	//O(N) algorithm
	if(step == 4)
	{
		for(int i=start; i<end;i++)
		{
			Vertex* leafNode = (*allLeaves)[i];
#ifdef METRICS
			leafNode->footprint++;
			numberOfNodesVisited++;
#endif
			if(leafNode->vData)
			{
				VertexData* vData = leafNode->vData->next;
				while(vData)
				{
					//double tmp=1.0;
					std::vector<Vertex*> neighbors;
					double x1 = vData->p->coordX;
					double y1 = vData->p->coordY;
					GetInteractionList(leafNode, neighbors, true);
					std::vector<Vertex*>::iterator nIter = neighbors.begin();
					for(;nIter!=neighbors.end();nIter++)
					{
						Vertex* otherNode=*nIter;
						VertexData* otherNodeVData = otherNode->vData->next;
						while(otherNodeVData)
						{
							double x2 = (otherNodeVData)->p->coordX;
							double y2 = (otherNodeVData)->p->coordY;
							vData->p->potential += KernelFn(x1, y1, x2, y2) * vData->p->mass;
							otherNodeVData = otherNodeVData->next;
						}
					}
					vData = vData->next;
				}
			}
		}
	}

}

void TraverseDown_Recursive(Vertex* node, int step)
{
	if(step == 3)
	{
		if(node->parent)
		{
			if(node->vData)
				node->vData->p->potential += node->parent->vData->p->potential;
		}
	}

	if(step == 2)
	{
		std::vector<Vertex*> wellSeparatedNodes;
		double x1 = (node->box.startX + node->box.endX) /(double) 2;
		double y1 = (node->box.startY + node->box.endY) /(double) 2;
		GetInteractionList(node, wellSeparatedNodes, false);
		std::vector<Vertex*>::iterator nIter = wellSeparatedNodes.begin();
		for(;nIter!=wellSeparatedNodes.end();nIter++)
		{
			double x2 = ((*nIter)->box.startX + (*nIter)->box.endX) /(double) 2;
			double y2 = ((*nIter)->box.startY + (*nIter)->box.endY) /(double) 2;
			if(node->vData)
				node->vData->p->potential += KernelFn(x1, y1, x2, y2) * node->vData->p->mass;
		}
	}

	if(node->isLeaf)
	{
		if(step == 3)
		{
			if(node->vData)
			{
				VertexData* vData = node->vData->next;
				while(vData)
				{
					vData->p->potential += node->vData->p->potential;
					vData = vData->next;
				}
			}
		}
	}
	else
	{
		for(int i=0;i<MAX_CHILDREN;i++)	
		{
			if(node->pChild[i])
			{
				TraverseDown_Recursive(node->pChild[i],step);
			}
		}
	}
	return;
}

void TraverseDownFused_Recursive(Vertex* node)
{
	std::vector<Vertex*> wellSeparatedNodes;
	double x1 = (node->box.startX + node->box.endX) /(double) 2;
	double y1 = (node->box.startY + node->box.endY) /(double) 2;
	GetInteractionList(node, wellSeparatedNodes, false);
	std::vector<Vertex*>::iterator nIter = wellSeparatedNodes.begin();
	for(;nIter!=wellSeparatedNodes.end();nIter++)
	{
		double x2 = ((*nIter)->box.startX + (*nIter)->box.endX) /(double) 2;
		double y2 = ((*nIter)->box.startY + (*nIter)->box.endY) /(double) 2;
		node->vData->p->potential += KernelFn(x1, y1, x2, y2) * node->vData->p->mass;	
	}

	if(node->parent)
		node->vData->p->potential += node->parent->vData->p->potential;

	if(node->isLeaf)
	{
		Vertex* leafNode = node;
		if(leafNode->vData)
		{
			VertexData* vData = leafNode->vData->next;
			while(vData)
			{
				vData->p->potential += leafNode->vData->p->potential;
				vData = vData->next;
			}
		}
	}
	else
	{
		for(int i=0;i<MAX_CHILDREN;i++)	
		{
			if(node->pChild[i])
			{
				TraverseDownFused_Recursive(node->pChild[i]);
			}
		}
	}
	return;
}

void GetInteractionList(Vertex* node, std::vector<Vertex*>& interactionList, bool neighbors)
{
	if(node->level < 2)
		return;
	if(neighbors)
		interactionList = node->neighbors;
	else
	{
		for(int i=0;i<node->neighbors.size();i++)
		{
			std::set<Vertex*> cellsConsidered;
			Vertex* neighbor = node->neighbors[i];
			if(neighbor->parent != node->parent)
			{
				Vertex* neighParent = neighbor->parent;
				std::pair<std::set<Vertex*>::iterator, bool> ret = cellsConsidered.insert(neighParent);
				if(ret.second)
				{
					for(int j=0;j<MAX_CHILDREN;j++)
					{
						if(neighParent->pChild[j] && !AreAdjacent(neighParent->pChild[j]->box,node->box))
						{
							interactionList.push_back(neighParent->pChild[j]);	
						}
					}
				}
			}
		}
	}
}

bool AreAdjacent(Box& b1, Box& b2)
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

double KernelFn(double x1, double y1, double x2, double y2)
{
	
	return log(sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)));
}

/*void random_particles(Point* particles, int m)
{
     int labelCount=0;
     for (int i =0 ; i < m ; i ++ )
       { 
       particles[i].coordX = (rand() % 128 ) ; 
       particles[i].coordY = (rand() % 128 ) ; 
       particles[i].mass = 2;
       particles[i].potential =0.;
       particles[i].id = labelCount++;
       }       
}*/

Box random_particles(Point* particles, int m)
{
	Box ret;
	if(particles)
     	{
		static int labelCount=0;
	     	for (int i =0 ; i < m ; i ++ )
       		{ 
	       		//particles[i].coordX = (rand() % 128 ) ; 
			//particles[i].coordY = (rand() % 128 ) ; 
	       		particles[i].coordX = (rand() /(double)RAND_MAX) ; 
			particles[i].coordY = (rand() /(double)RAND_MAX) ; 

			if(particles[i].coordX < ret.startX)
				ret.startX = particles[i].coordX;
			if(particles[i].coordX > ret.endX)
				ret.endX = particles[i].coordX;
			if(particles[i].coordY < ret.startY)
				ret.startY = particles[i].coordY;
			if(particles[i].coordY > ret.endY)
				ret.endY = particles[i].coordY;
			ret.startX = floor(ret.startX);
			ret.endX = ceil(ret.endX);
			ret.startY = floor(ret.startY);
			ret.endY = ceil(ret.endY);

			particles[i].mass = 2;
			particles[i].potential =0.;
			particles[i].id = labelCount++;
	       }
	}
	else
	{
		for (int i =0 ; i < m ; i ++ )
		{
			/*int coordX = (rand() % 128 ) ; 
			int coordY = (rand() % 128 ) ; */
	       		double coordX = (rand() /(double)RAND_MAX) ; 
			double coordY = (rand() /(double)RAND_MAX) ; 

			if(coordX < ret.startX)
				ret.startX = coordX;
			if(coordX > ret.endX)
				ret.endX = coordX;
			if(coordY < ret.startY)
				ret.startY = coordY;
			if(coordY > ret.endY)
				ret.endY = coordY;

			ret.startX = floor(ret.startX);
			ret.endX = ceil(ret.endX);
			ret.startY = floor(ret.startY);
			ret.endY = ceil(ret.endY);
		}
	}

	return ret;
	       
}





void parallel_for(thread_function func, int start, int end, void* inParam, void* outParam)
{
	int numTraversals = end - start;
	if (numThreads == 1) 
	{
		func(inParam, start, end, outParam);
	} 
	else 
	{

		pthread_t *threads = NULL;
		threads = new pthread_t[numThreads];
		if(!threads) 
		{
			fprintf(stderr, "error: could not allocate threads.\n");
			exit(1);
		}
		struct targs *args = new struct targs[numThreads];
		if(!args) 
		{
			fprintf(stderr, "error: could not allocate thread args.\n");
			exit(1);
		}
		int *ranges = distribute_among(0, numTraversals, numThreads);
		for(int i = 0; i < numThreads; i++) 
		{
			args[i].start = ranges[2 * i];
			args[i].end = ranges[2 * i + 1];
			args[i].func = func;
			args[i].funcIn = inParam;
			args[i].funcOut = outParam;
			if(pthread_create(&threads[i], NULL, thread_entry, &args[i]) != 0)
			{
				fprintf(stderr, "error: could not create thread.\n");
			}
		}
		delete [] ranges;

		for(int i = 0; i < numThreads; i++) 
		{
			pthread_join(threads[i], NULL);
		}
		
		delete [] threads;
	}
}

int main(int argc, char** argv)
{
	double startTime, endTime;
	if((argc > 4)||(argc < 2))
	{
		printf("usage: ./FMM <num_points> <num_threads(optional)>\n");
		return 0;
	}
	long int numPoints = atol(argv[1]);
	long int batchSize = 10000;
	std::vector<Point*> inputPointerArray;
#ifdef DEBUG
	std::vector<int> inputArraySizes;
#endif
	Point* pointArr=NULL;
	Box box;
	srand(0);
	if(argc == 4)
	{
		read_input(argv[3],pointArr,numPoints);
		numThreads = atoi(argv[2]);
		printf("Params: num points %ld num threads %d\n",numPoints, numThreads); 
	}
	else
	{
		if(argc == 3)
			numThreads = atoi(argv[2]);
		printf("Params: num points %ld num threads %d input file None\n",numPoints, numThreads); 
	}

	if(pthread_mutex_init(&fmm_mutex,NULL))
		printf("ERROR: synchronization among threads will not work\n"); 
	
	srand(0);
	/*if(numPoints > batchSize)
	{
		long int tmpPoints = 0;
		while(tmpPoints < numPoints)
		{
			int curBatchSize = ((tmpPoints+batchSize)<numPoints)?batchSize:(numPoints - tmpPoints);
			pointArr = new Point[curBatchSize];
			random_particles(pointArr,curBatchSize);
			tmpPoints += curBatchSize;		
			bool hasMoreData = (tmpPoints < numPoints)?true:false;
			ConstructQuadTreeAndTraverse(pointArr, box, curBatchSize, hasMoreData);
			inputPointerArray.push_back(pointArr);
#ifdef DEBUG
			inputArraySizes.push_back(curBatchSize);
#endif
		}
	}
	else*/
	{
		pointArr = new Point[numPoints];
		box = random_particles(pointArr,numPoints);
		double boxHeight = box.endY - box.startY;
		double boxWidth = box.endX - box.startX;
		if(boxWidth > boxHeight)
			box.endY = box.startY + boxWidth;
		else
			box.endX = box.startX + boxHeight;


		ConstructQuadTreeAndTraverse(pointArr, box, numPoints, false);
		delete [] pointArr;
	}
	
#ifdef DEBUG
	/*FILE* fp=fopen("pointslog.txt","w");
	for(int i=0;i<inputPointerArray.size();i++)
	{
		Point* pointArr = inputPointerArray[i];
		for(int j=0;j<inputArraySizes[i];j++)
		{
			fprintf(fp,"%d %f\n",pointArr[j].id, pointArr[j].potential);
		}
	}
	fclose(fp);*/
#endif
#ifdef METRICS
	printf("Number of vertices %d\n",numberOfVertices);
	printf("Total number of traversals %d\n",numberOfTraversals);
	printf("Total number of nodes visited %d\n",numberOfNodesVisited);
	printLoadDistribution();
#endif
	for(int i=0;i<inputPointerArray.size();i++)
		delete [] inputPointerArray[i];
	return 0;
}


void read_input(char* inputFile, Point* pointArr, long int numPoints) {
	long int i, j, k;
	double min = FLT_MAX;
	double max = -FLT_MAX;
	FILE *in;

	if(inputFile != NULL) 
	{
		in = fopen(inputFile, "r");
		if(in == NULL) {
			fprintf(stderr, "Could not open %s\n", inputFile);
			exit(1);
		}

		for(i = 0; i < numPoints; i++) 
		{
			read_point(in, &pointArr[i], i);
		}
		fclose(in);

	} 
	else 
	{
		random_particles(pointArr,numPoints);
	}
}

void read_point(FILE *in, Point *p, long int label) 
{
	int j;
	if(fscanf(in, "%f", &p->coordX) != 1) {
		fprintf(stderr, "Input file not large enough.\n");
		exit(1);
	}
	if(fscanf(in, "%f", &p->coordY) != 1) {
		fprintf(stderr, "Input file not large enough.\n");
		exit(1);
	}
	p->mass = 2;
	p->potential = 0;
	p->id = label;
}

int ComputeChildNumber(const Vec& cofm, const Point* child)
{
	int numChild = 0;
	if(cofm.pt[0] < child->coordX)
	{
		numChild = 1;
	}
	if(cofm.pt[1] < child->coordY) 
	{
		numChild += 2;
	}
	return numChild;
}

void ComputeBoundingBoxParams(Box& box, Vec& center, double& dia)
{
	center.pt[0] = (box.startX + box.endX)/2;
	center.pt[1] = (box.startY + box.endY)/2;
	dia = box.endX - box.startX;
	return;
}
/*void ComputeBoundingBoxParams(const Point* nbodies, long int numPoints, Vec& center, double& dia)
{
	Vec max(-FLT_MAX), min(FLT_MAX);

	for (int i=0; i<numPoints; i ++ ) 
	{
		const Point* p = &(nbodies[i]);
		for(int i=0;i<DIMENSION;i++)
		{
			if(i == 0)
			{
				if(min.pt[i] > p->coordX)
					min.pt[i] = p->coordX;
				if(max.pt[i] < p->coordX)
					max.pt[i] = p->coordX;
			}
			if(i == 1)
			{
				if(min.pt[i] > p->coordY)
					min.pt[i] = p->coordY;
				if(max.pt[i] < p->coordY)
					max.pt[i] = p->coordY;
			}
		}
	}

	dia = max.pt[0] - min.pt[0];
	for(int i=1;i<DIMENSION;i++)
	{
		if (dia<(max.pt[i]-min.pt[i]))
		{
			dia=max.pt[i]-min.pt[i];
		}
	}

	for(int i=0;i<DIMENSION;i++)
	{
		center.pt[i] = (max.pt[i] + min.pt[i])/2;
	}
	return;

}*/

void GetLeafNodes(Vertex* node)
{
	if(node->isLeaf)
	{
		allLeaves.push_back(node);
		return;
	}
	
	for(int i=0;i<MAX_CHILDREN;i++)
	{
		if(node->pChild[i])
			GetLeafNodes(node->pChild[i]);
	}
}

#ifdef METRICS
void printLoadDistribution()
{
	printf("num bottom subtrees %d\n",subtrees.size());
	std::vector<Vertex*>::iterator iter = subtrees.begin();
	for(;iter != subtrees.end();iter++)
	{
		long int num_vertices=0, footprint=0;
		subtreeStats stats;
		getSubtreeStats(*iter, &stats);
		printf("id %d (%p) num_vertices %d footprint %ld\n",(*iter)->label, *iter, stats.numnodes, stats.footprint);
	}
}

void getSubtreeStats(Vertex* ver, subtreeStats* stats)
{
		stats->numnodes += 1;
		stats->footprint += ver->footprint;
		assert(ver != NULL);

		for(int i=0;i<MAX_CHILDREN;i++)
		{
			if(ver->pChild[i])
			{
				getSubtreeStats(ver->pChild[i],stats);
			}
		}
}


#endif

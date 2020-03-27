#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
using std::vector;

#define N 512 
#define THREADS_PER_BLOCK 32
#define m 1000000 


struct particle { 
    short X;
    short Y;
    int mass;
    int potential;
};
struct particle particles[m];


struct node {  
    short startX;
    short midX;
    short endX;
    short startY;
    short midY;
    short endY;
    short width;
    short height;
    short parentIndex;
    short myLevel;
    int  mass;
    int potential;
    vector<int> mychild;
    vector<particle> mypoints;
    vector<int> interactionList; 

}; 
 
struct GPUNode {  
    short startX; 
    short midX; 
    short endX;
    short startY;
    short midY;
    short endY;
    short parentIndex;
    short myLevel; 
    int  mass;
    int potential; 
    int mychild[4];
    int interactionList[27]; 
    bool done;

}; 


struct node setRootNode( short sx, short mx , short ex, short sy ,  short my  , short ey, short width , short height , int massIn , int potentialIn) { 
  node newNode;    
  newNode.startX = sx;
  newNode.endX = ex;
  newNode.startY = sy;
  newNode.endY = ey;
  newNode.midX = mx;
  newNode.midY = my;
  newNode.width = width;
  newNode.height = height;
  newNode.parentIndex = 0 ; 
  newNode.mass = massIn;
  newNode.potential = potentialIn; 
  newNode.myLevel = 0 ;

  for ( int i = 0 ; i < m ; i++)
   {
    newNode.mypoints.push_back(particles[i]);
   }
   printf("size of point space %d\n" , newNode.mypoints.size()) ;
  return(newNode); 
} 

//-------------------------------------
// GENERATE RANDOM PARTICLES
//-------------------------------------
void random_particles (struct particle particles[m])
{
     for (int i =0 ; i < m ; i ++ )
       { 
       particles[i].X = (rand() % 128 ) ; 
       particles[i].Y = (rand() % 128 ) ; 
       particles[i].mass = 2;
       particles[i].potential =0 ;
       }       
}


//-------------------------------------
// PRINT PARTICLE
//-------------------------------------
void printParticle(struct particle k){
printf(" X: %d \t Y: %d \t mass: %d \t  potential: %d \n",k.X, k.Y, k.mass,  k.potential);
printf("\n");
}
 


//-------------------------------------
// PRINT PROBLEM
//-------------------------------------
void printProblem(struct particle particles[m]) 
{
int i ;
for ( i = 0 ; i < m ; i++)
   {
   printf("printing particle : %d \n" ,  i );
   printParticle(particles[i]);
   }
}


//-------------------------------------
// PRINT QUAD TREE
//-------------------------------------
void PrintQuadTree(vector<node> n, int depth)
{
         
    for (int i = 0 ; i < n.size() ;  i++ )
        {
         printf("  sx: %d , ex: %d, sy: %d , ey: %d  , numPoints: %d \n",  n[i].startX , n[i].endX , n[i].startY , n[i].endY  , n[i].mypoints.size() ) ;
        }
}

void printBox(node n)
{
   printf("************************ \n");
   printf("n->startX : %d n->endX: %d\n" , n.startX , n.endX);
   printf("n->startY : %d n->endY: %d\n" , n.startY , n.endY);
   printf("n->midX : %d n->midY: %d\n" , n.midX , n.midY);
   printf("width : %d height: %d\n" , n.width , n.height);
   printf("************************ \n");
}


//-------------------------------------
// BUILD NODE
//-------------------------------------
struct node BuildNode(vector<node> n , int index, int level , int maxLevel , int fakeparent_index)
{
    int  i = 0;

    int remainingLevels = maxLevel - level ;
    vector<int> ll;
    for (int i =0 ; i < remainingLevels ; i ++)
        {
        ll.push_back(pow(4,i+1)); 
        }
    
 
    int ind = n.size() -1 ; 

    for (int i =0 ; i < ll.size(); i++)
         {
          ind = ind - index * ll[i];
         }
    ind = ind-index;




    node newNode; 
        newNode.parentIndex= ind;   



    newNode.myLevel=level ;
    newNode.width = n[ind].width/2;
    newNode.height = n[ind].height/2;
    switch(index)
    {
        case 0: // NE
           newNode.startX = n[ind].startX;
           newNode.startY = n[ind].startY;
            break;
        case 1: // NW
            newNode.startX = n[ind].startX;
            newNode.startY = n[ind].startY+ newNode.width ;
            break;             
        case 2: // SW             
            newNode.startX = n[ind].startX+ newNode.width ;
            newNode.startY = n[ind].startY;
           break;
        case 3: // SE
            newNode.startX = n[ind].startX + newNode.width ;
            newNode.startY = n[ind].startY+ newNode.width ;
            break;
    }

    newNode.endX =  newNode.startX + newNode.width  ; 
    newNode.endY =  newNode.startY + newNode.height ; 
    newNode.midX =  newNode.startX + (newNode.width)/2 ; 
    newNode.midY =  newNode.startY + (newNode.height)/2 ;
    newNode.mass = 0;
    newNode.potential = 0; 

    for(i = 0; i <  n[ind].mypoints.size(); i++)
            { 

                  if ( n[ind].mypoints.at(i).X >= newNode.startX &&  n[ind].mypoints.at(i).X <= (newNode.endX-1) &&  n[ind].mypoints.at(i).Y >= newNode.startY &&  n[ind].mypoints.at(i).Y <= (newNode.endY-1) )
                     {


                     newNode.mypoints.push_back(n[ind].mypoints.at(i));
                     }
            }  


    return newNode;
}


//-------------------------------------
// BUILD QUAD TREE
//-------------------------------------
void BuildQuadTree(vector<node>  &n , short level, short maxLevel)
{    


    int parent_index = n.size()-1; 
    if(level < maxLevel)  
    {

           level++;
        for(int k =0; k < 4; k++)
        { 
            node newNode = BuildNode(n , k , level , maxLevel , parent_index); 
            n.push_back(newNode);
            int parentIndex =  newNode.parentIndex;
            int myIndex =  n.size()-1;

            n[parentIndex].mychild.push_back(myIndex); 
            BuildQuadTree( n , level , maxLevel );
        }

    }
}


//-------------------------------------
// FMM step 1
//-------------------------------------
int FMMS1 (vector<node>  &n , short maxLevel)
{
    
    for (int j =0 ; j < n.size(); j++)
        {
         if (n[j].myLevel == maxLevel)
            {
             for (int i =0 ; i < n[j].mypoints.size() ; i++)
                 {
                  n[j].mass += n[j].mypoints[i].mass;
                 }
            }
        }
   short currentLevel =maxLevel -1 ; 
   while (currentLevel !=-1)
         {
         for (int j =0 ; j < n.size(); j++)
            {
            if (n[j].myLevel == currentLevel) 
               {
                for (int z =0 ; z < n[j].mychild.size() ; z++)
                    {
                    n[j].mass += n[n[j].mychild[z]].mass;
                    } 
               }
             }
           currentLevel--;
          }
    
    return 1;
}

//-------------------------------------
// FMM step 2
//-------------------------------------
int FMMS2 (vector<node>  &n , short maxLevel)
{

for (int i =0 ; i < n.size(); i++) 
    {
    if (n[i].myLevel > 2)
       {
       node parent = n[n[i].parentIndex];


       int parentIndex = n[i].parentIndex;   

       node grandParent = n[parent.parentIndex];  


       node greatGrandParent = n[grandParent.parentIndex];    


       int a[4], b[16]; 
       vector<int> parentNeighbors ; 
       for (int j =0 ; j < 4; j++)
           {
           a[j]= greatGrandParent.mychild[j]; 

           }
       for (int j =0 , z=0 ; z< 16 && j <4 ; j++ )
          {
           b[z] = n[a[j]].mychild[0]; 
           b[z+1] = n[a[j]].mychild[1];
           b[z+2] = n[a[j]].mychild[2];
           b[z+3] = n[a[j]].mychild[3];
           z+=4;
          }
      for (int j =0 ; j < 16 ; j ++)
          {
           if (parentIndex != b[j])
              {

               if (n[parentIndex].startX==n[b[j]].endX && n[parentIndex].startY==n[b[j]].endY) 
                   {parentNeighbors.push_back(b[j]); 

                     }
               else if (n[parentIndex].startX==n[b[j]].endX && n[parentIndex].startY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                     }
               else if (n[parentIndex].endX==n[b[j]].startX && n[parentIndex].startY==n[b[j]].endY)
                   {parentNeighbors.push_back(b[j]); 

                     }
               else if (n[parentIndex].startX==n[b[j]].endX && n[parentIndex].endY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                    }
               else if (n[parentIndex].endX==n[b[j]].startX && n[parentIndex].startY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                   }
               else if (n[parentIndex].startX==n[b[j]].startX && n[parentIndex].endY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                    }
               else if (n[parentIndex].startX==n[b[j]].startX && n[parentIndex].startY==n[b[j]].endY)
                   {parentNeighbors.push_back(b[j]); 

                    }
               else if (n[parentIndex].endX==n[b[j]].startX && n[parentIndex].endY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                   }
              }
          }

     for (int j =0 ; j < parentNeighbors.size() ; j ++)
        {
        for (int z =0 ; z <4 ; z ++ )
            {
             bool neighbor = false ; 
             if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY) 
                {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].startY)
                {neighbor = true ; }
             else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)
                  {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)
                  {neighbor = true ; }
             else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             if (!neighbor)
                {

                 n[i].potential = n[i].potential + sqrt( (float)(n[n[parentNeighbors[j]].mychild[z]].midX-n[i].midX)* (n[n[parentNeighbors[j]].mychild[z]].midX-n[i].midX) + (n[n[parentNeighbors[j]].mychild[z]].midY-n[i].midY)* (n[n[parentNeighbors[j]].mychild[z]].midY-n[i].midY) )*  n[n[parentNeighbors[j]].mychild[z]].mass;
                }
            }  
        }     
     }
         if (n[i].myLevel == 2)
            {
                   int parentIndex = n[i].parentIndex;    
                   int grandParentIndex = n[parentIndex].parentIndex;
                   vector <int> parentNeighbors ;
                   for (int j =0 ; j < 4 ; j ++)
                       {
                       if (n[grandParentIndex].mychild[j] != parentIndex )
                           parentNeighbors.push_back(n[grandParentIndex].mychild[j]);
                        }  
                  for(int j =0 ; j < parentNeighbors.size() ; j++)
                     {
 		     for (int z =0 ; z <4 ; z ++ )
                         {
                         bool neighbor = false ; 
                         if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY) 
                            {neighbor = true ; }
                        else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].startY)
                            {neighbor = true ; }
                        else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)
                            {neighbor = true ; }
                        else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                            {neighbor = true ; }
                        else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].startY)
                            {neighbor = true ; }
                        else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                            {neighbor = true ; }
                        else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)
                            {neighbor = true ; }
                        else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                            {neighbor = true ; }
                        if (!neighbor)    
                           {

                            n[i].potential = n[i].potential + sqrt( (float) (n[n[parentNeighbors[j]].mychild[z]].midX-n[i].midX)* (n[n[parentNeighbors[j]].mychild[z]].midX-n[i].midX) + (n[n[parentNeighbors[j]].mychild[z]].midY-n[i].midY)* (n[n[parentNeighbors[j]].mychild[z]].midY-n[i].midY) )*  n[n[parentNeighbors[j]].mychild[z]].mass;
                           }
                        }  
                     }
            }

   }
          return 0;
}


//-------------------------------------
// FMM step 2 part a interaction list computation
//-------------------------------------
int interactionListCreation (vector<node>  &n , short maxLevel)
{

for (int i =0 ; i < n.size(); i++) 
    {
    if (n[i].myLevel > 2)
       {
       node parent = n[n[i].parentIndex];


       int parentIndex = n[i].parentIndex;   

       node grandParent = n[parent.parentIndex]; 


       node greatGrandParent = n[grandParent.parentIndex];  


       int a[4], b[16]; 
       vector<int> parentNeighbors ; 
       for (int j =0 ; j < 4; j++)
           {
           a[j]= greatGrandParent.mychild[j]; 

           }
       for (int j =0 , z=0 ; z< 16 && j <4 ; j++ )
          {
           b[z] = n[a[j]].mychild[0]; 
           b[z+1] = n[a[j]].mychild[1];
           b[z+2] = n[a[j]].mychild[2];
           b[z+3] = n[a[j]].mychild[3];
           z+=4;
          }
      for (int j =0 ; j < 16 ; j ++)
          {
           if (parentIndex != b[j])
              {

               if (n[parentIndex].startX==n[b[j]].endX && n[parentIndex].startY==n[b[j]].endY)  
                   {parentNeighbors.push_back(b[j]); 

                     }
               else if (n[parentIndex].startX==n[b[j]].endX && n[parentIndex].startY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                     }
               else if (n[parentIndex].endX==n[b[j]].startX && n[parentIndex].startY==n[b[j]].endY)
                   {parentNeighbors.push_back(b[j]); 

                     }
               else if (n[parentIndex].startX==n[b[j]].endX && n[parentIndex].endY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                    }
               else if (n[parentIndex].endX==n[b[j]].startX && n[parentIndex].startY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                   }
               else if (n[parentIndex].startX==n[b[j]].startX && n[parentIndex].endY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                    }
               else if (n[parentIndex].startX==n[b[j]].startX && n[parentIndex].startY==n[b[j]].endY)
                   {parentNeighbors.push_back(b[j]); 

                    }
               else if (n[parentIndex].endX==n[b[j]].startX && n[parentIndex].endY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                   }
              }
          }

     for (int j =0 ; j < parentNeighbors.size() ; j ++)
        {
        for (int z =0 ; z <4 ; z ++ )
            {
             bool neighbor = false ; 
             if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY) 
                {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].startY)
                {neighbor = true ; }
             else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)
                  {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)
                  {neighbor = true ; }
             else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             if (!neighbor)  
                {

                 n[i].interactionList.push_back(n[parentNeighbors[j]].mychild[z]);
                }
            }  
        }     
     }
         if (n[i].myLevel == 2)
            {
                   int parentIndex = n[i].parentIndex;    
                   int grandParentIndex = n[parentIndex].parentIndex;
                   vector <int> parentNeighbors ;
                   for (int j =0 ; j < 4 ; j ++)
                       {
                       if (n[grandParentIndex].mychild[j] != parentIndex )
                           parentNeighbors.push_back(n[grandParentIndex].mychild[j]);
                        }  
                  for(int j =0 ; j < parentNeighbors.size() ; j++)
                     {
 		     for (int z =0 ; z <4 ; z ++ )
                         {
                     
                         bool neighbor = false ; 
                         if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)  
                            {neighbor = true ; }
                        else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].startY)
                            {neighbor = true ; }
                        else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)
                            {neighbor = true ; }
                        else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                            {neighbor = true ; }
                        else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].startY)
                            {neighbor = true ; }
                        else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                            {neighbor = true ; }
                        else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)
                            {neighbor = true ; }
                        else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                            {neighbor = true ; }
                        if (!neighbor)   
                           {

                           n[i].interactionList.push_back(n[parentNeighbors[j]].mychild[z]);
                           }
                        }  
                     }
            }

   }
          return 0;
}

//-------------------------------------
// FMM step 3
//-------------------------------------
int FMMS3 (vector<node>  &n , short maxLevel)
{
for ( int i =0 ; i < n.size(); i ++ )
    {
     if (n[i].myLevel == maxLevel)    
        {
         for (int j =0; j < n[i].mypoints.size(); j++)
             {
              n[i].mypoints[j].potential =  n[i].mypoints[j].potential + n[i].potential;
              int index = n[i].parentIndex;
              while( index !=0)
                   {
                   n[i].mypoints[j].potential =  n[i].mypoints[j].potential + n[index].potential;

                   index=  n[index].parentIndex;
                   }
             }
         }
    }   
    return 1;
}


//-------------------------------------
// FMM step 4
//-------------------------------------
int FMMS4 (vector<node>  &n , short maxLevel)
{
for ( int i =0 ; i < n.size(); i ++ )
    {
     if (n[i].myLevel == maxLevel)    
         {
       node parent = n[n[i].parentIndex];
      
       int parentIndex = n[i].parentIndex;    

       node grandParent = n[parent.parentIndex]; 


       node greatGrandParent = n[grandParent.parentIndex];  


       int a[4], b[16]; 
       vector<int> parentNeighbors ; 
       for (int j =0 ; j < 4; j++)
           {
           a[j]= greatGrandParent.mychild[j]; 
           }
       for (int j =0 , z=0 ; z< 16 && j <4 ; j++ )
          {
           b[z] = n[a[j]].mychild[0]; 
           b[z+1] = n[a[j]].mychild[1];
           b[z+2] = n[a[j]].mychild[2];
           b[z+3] = n[a[j]].mychild[3];
           z+=4;
          }

      parentNeighbors.push_back(parentIndex); 
      for (int j =0 ; j < 16 ; j ++)
          {
           
               if (n[parentIndex].startX==n[b[j]].endX && n[parentIndex].startY==n[b[j]].endY) 
                   {parentNeighbors.push_back(b[j]); 

                     }
               else if (n[parentIndex].startX==n[b[j]].endX && n[parentIndex].startY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                     }
               else if (n[parentIndex].endX==n[b[j]].startX && n[parentIndex].startY==n[b[j]].endY)
                   {parentNeighbors.push_back(b[j]); 

                     }
               else if (n[parentIndex].startX==n[b[j]].endX && n[parentIndex].endY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                    }
               else if (n[parentIndex].endX==n[b[j]].startX && n[parentIndex].startY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                   }
               else if (n[parentIndex].startX==n[b[j]].startX && n[parentIndex].endY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                    }
               else if (n[parentIndex].startX==n[b[j]].startX && n[parentIndex].startY==n[b[j]].endY)
                   {parentNeighbors.push_back(b[j]); 

                    }
               else if (n[parentIndex].endX==n[b[j]].startX && n[parentIndex].endY==n[b[j]].startY)
                   {parentNeighbors.push_back(b[j]); 

                   }
           
          }

     for (int j =0 ; j < parentNeighbors.size() ; j ++)
        {
        for (int z =0 ; z <4 ; z ++ )
            {
             bool neighbor = false ; 
             if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY) 
                {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].startY)
                {neighbor = true ; }
             else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)
                  {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].endX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             else if (n[i].startX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].startY==n[n[parentNeighbors[j]].mychild[z]].endY)
                  {neighbor = true ; }
             else if (n[i].endX==n[n[parentNeighbors[j]].mychild[z]].startX && n[i].endY==n[n[parentNeighbors[j]].mychild[z]].startY)
                  {neighbor = true ; }
             if (neighbor)   
                {

                for (int t =0 ; t < n[i].mypoints.size(); t++)
                     {

                      for (int v =0 ; v <n[n[parentNeighbors[j]].mychild[z]].mypoints.size() ; v++ )
                          {
                          n[i].mypoints[t].potential = n[i].mypoints[t].potential +  sqrt( pow ( (n[n[parentNeighbors[j]].mychild[z]].mypoints[v].X - n[i].mypoints[t].X) ,2)  + pow ( (n[n[parentNeighbors[j]].mychild[z]].mypoints[v].Y - n[i].mypoints[t].Y) ,2)   )* n[n[parentNeighbors[j]].mychild[z]].mypoints[v].mass ;  
                          } 

                     }
                }
            }  
        }     
     }
    }   
    return 1;
}


//-------------------------------------
// FMM step 2 GPU
//-------------------------------------
__global__ void FMM2( GPUNode * n , short schedule[171][128]) 

{
  int i =schedule[blockIdx.x][4*threadIdx.x];
  int i2 =schedule[blockIdx.x][4*threadIdx.x+1];
  int i3 =schedule[blockIdx.x][4*threadIdx.x+2];
  int i4 =schedule[blockIdx.x][4*threadIdx.x+3]; 
 if(i != -1 )
  {
   for( int j =0 ; j < 27; j++)
      {
      if(n[i].interactionList[j]!= -1 )
        { 
         n[i].potential = n[i].potential + sqrt( (float) (n[n[i].interactionList[j]].midX-n[i].midX)* (n[n[i].interactionList[j]].midX-n[i].midX) + (n[n[i].interactionList[j]].midY-n[i].midY)* (n[n[i].interactionList[j]].midY-n[i].midY) )*  n[n[i].interactionList[j]].mass;
           n[i].done= true;
        }
     }
  }
if(i2 != -1 )
  {
 for( int j =0 ; j < 27; j++)
      {      
      if(n[i2].interactionList[j]!= -1 )
        { 
         n[i2].potential = n[i2].potential + sqrt( (float) (n[n[i2].interactionList[j]].midX-n[i2].midX)* (n[n[i2].interactionList[j]].midX-n[i2].midX) + (n[n[i2].interactionList[j]].midY-n[i2].midY)* (n[n[i2].interactionList[j]].midY-n[i2].midY) )*  n[n[i2].interactionList[j]].mass;
         n[i2].done= true;
        }
    }
  }
if(i3 != -1 )
  {
   for( int j =0 ; j < 27; j++)
      {
      if(n[i3].interactionList[j]!= -1 )
        { 
         n[i3].potential = n[i3].potential + sqrt( (float) (n[n[i3].interactionList[j]].midX-n[i3].midX)* (n[n[i3].interactionList[j]].midX-n[i3].midX) + (n[n[i3].interactionList[j]].midY-n[i3].midY)* (n[n[i3].interactionList[j]].midY-n[i3].midY) )*  n[n[i3].interactionList[j]].mass;
        n[i3].done= true;
        }
     }
   }
if(i4 != -1 )
  {
   for( int j =0 ; j < 27; j++)
      {
      if(n[i4].interactionList[j]!= -1 )
        { 
         n[i4].potential = n[i4].potential + sqrt( (float) (n[n[i4].interactionList[j]].midX-n[i4].midX)* (n[n[i4].interactionList[j]].midX-n[i4].midX) + (n[n[i4].interactionList[j]].midY-n[i4].midY)* (n[n[i4].interactionList[j]].midY-n[i4].midY) )*  n[n[i4].interactionList[j]].mass;
        n[i4].done= true;
        }
      } 
  }
}


void shuffleIndices(short random_index[171][128], vector<node>  &n)
{
int counter[5];
int level4Counter = 1; 
int level5Counter = 3; 
int level6Counter = 11; 
int level7Counter = 43; 
for (int t =0 ; t <n.size(); t++ )
{
if (n[t].myLevel <= 3 )
  {
  random_index[0][counter[0]] = t ;
  counter[0]++;
  }
if (n[t].myLevel == 4 )
  {
  random_index[level4Counter][counter[1]] = t ;
  counter[1]++;
  if (counter[1]==128)
     {
     counter[1]=0;
     level4Counter++;
     }
  }
if (n[t].myLevel ==5 )
  {
  random_index[level5Counter][counter[2]] = t ;
  counter[2]++;
  if (counter[2]==128)
     {

     counter[2]=0;
     level5Counter++;
     }
  }
if (n[t].myLevel ==6 )
  {
  random_index[level6Counter][counter[3]] = t ;
  counter[3]++;
  if (counter[3]==128)
     {
     counter[3]=0;
     level6Counter++;
     }
  }
if (n[t].myLevel ==7 )
  {
  random_index[level7Counter][counter[4]] = t ;
  counter[4]++;
  if (counter[4]==128)
     {
     counter[4]=0;
     level7Counter++;
     }
  }
}
for (int i = 85; i < 128 ; i ++)
      random_index[0][i] = -1 ;
}



void shuffle(short *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          short t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}


void  randomize(short random_index[171][128] )
{
short arrayCopy [21888];
  
  for(int i =0; i < 171 ; i++)
     {
     for (int j =0 ; j < 128 ; j++)
         {
         arrayCopy[i*128+j] = random_index[i][j];
         }
     }
 
shuffle(arrayCopy , 21888 ) ; 

  for(int i =0; i < 171 ; i++)
     {
     for (int j =0 ; j < 128 ; j++)
         {
          random_index[i][j] = arrayCopy[i*128+j] ;
         }
     }


}
//-------------------------------------
// MAIN
//-------------------------------------
int main ()
{
short maxLevel = 7  ;
random_particles(particles);
vector<node> myNodes;
myNodes.push_back( setRootNode( 0, 64 , 128 , 0 , 64 , 128, 128, 128 , 0 , 0) );
printf("building QuadTree\n");
BuildQuadTree(myNodes , 0 , maxLevel );

printf("FMM step 1 on CPU \n");
int totalMass=FMMS1(myNodes, maxLevel);

interactionListCreation (myNodes ,  maxLevel);

GPUNode GPUNodes[myNodes.size()];
for (int i = 0 ; i < myNodes.size(); i++)
    {
     GPUNodes[i].startX =  myNodes[i].startX ;
     GPUNodes[i].midX =  myNodes[i].midX ;
     GPUNodes[i].endX =  myNodes[i].endX ;
     GPUNodes[i].startY =  myNodes[i].startY ;
     GPUNodes[i].midY =  myNodes[i].midY ;
     GPUNodes[i].endY =  myNodes[i].endY ;
     GPUNodes[i].parentIndex =  myNodes[i].parentIndex ;  
     GPUNodes[i].myLevel =  myNodes[i].myLevel ; 
     GPUNodes[i].mass =  myNodes[i].mass ;
     GPUNodes[i].potential =  myNodes[i].potential ; 
       GPUNodes[i].done =  false ; 
     if (myNodes[i].mychild.size() !=0)
        {
         for(int j = 0 ; j<4 ; j++ )
             GPUNodes[i].mychild[j] =  myNodes[i].mychild[j] ;  
         }
     else
        {
        for(int j = 0 ; j<4 ; j++)
         GPUNodes[i].mychild[j] = 0 ;  
        }
     for(int j = 0 ; j<27 ; j++)
         GPUNodes[i].interactionList[j] = -1 ; 

     if (myNodes[i].interactionList.size() !=0) 
        {
         for(int j = 0 ; j< myNodes[i].interactionList.size() ; j++ )
             GPUNodes[i].interactionList[j] =  myNodes[i].interactionList[j] ;  
         }
    }

  short schedule[171][128] ;
  shuffleIndices(schedule, myNodes);
  int GPUNodeSize = sizeof(GPUNode);
  GPUNode *GPUNodesArray = new GPUNode[myNodes.size()];
  short  *GPUschedule;
  int size = sizeof(short ) * 171 * 128 ;
   cudaEvent_t start, stop;
  printf("FMM step 2 on GPU \n");
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMalloc((void**)&GPUNodesArray,myNodes.size()* GPUNodeSize);
  cudaMemcpy (GPUNodesArray, GPUNodes, myNodes.size()* GPUNodeSize , cudaMemcpyHostToDevice );
  cudaMalloc((void**)&GPUschedule, sizeof(int ) * 171 * 128);
  cudaEventRecord(start);
  cudaMemcpy (GPUschedule, &schedule,size ,cudaMemcpyHostToDevice);   
  FMM2<<<171 , 32 >>>( GPUNodesArray, (short (*)[128]) GPUschedule ); 
  cudaEventSynchronize(stop);
  cudaMemcpy (GPUNodes , GPUNodesArray ,myNodes.size()* GPUNodeSize  , cudaMemcpyDeviceToHost );
  cudaEventRecord(stop);

   printf("FMM step 3 on CPU \n");
  FMMS3(myNodes, maxLevel);
  printf("FMM step 4 on CPU \n");
  FMMS4(myNodes, maxLevel);
  if (cudaGetLastError() != cudaSuccess)
	printf("kernel launch failed\n");
  cudaThreadSynchronize();

  if (cudaGetLastError() != cudaSuccess)
	printf("kernel execution failed\n");
return 0; 
}

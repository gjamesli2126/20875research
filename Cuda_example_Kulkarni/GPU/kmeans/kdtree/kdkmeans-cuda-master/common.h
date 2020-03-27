/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef COMMON_H
#define COMMON_H

//#define N 1024
#define N 1000 * 200
//#define K 8   
#define K 100

#ifdef DEBUG
#define NDEBUG
#endif

enum {
    X_AXIS,
    Y_AXIS,
    DIMENSIONS,
};

#define SQUARE(x) ((x)*(x))
#define MOD(x) (((x) >= 0) ? (x) : (-(x)))

typedef struct {
    double loc[DIMENSIONS];
    int clusterId;
} Point;

typedef struct {
    Point pt; // Centroid+clusterId
    unsigned int noOfPoints;
} Cluster;


static double GetDistance(Point p1, Point p2) 
{
    return sqrt(SQUARE(p2.loc[X_AXIS]-p1.loc[X_AXIS])+SQUARE(p2.loc[Y_AXIS]-p1.loc[Y_AXIS]));
}

#if defined (GPU) || defined (GPU_KD)
static __device__ double GetDistanceGPU(Point p1, Point p2) 
{
    return sqrt(SQUARE(p2.loc[X_AXIS]-p1.loc[X_AXIS])+SQUARE(p2.loc[Y_AXIS]-p1.loc[Y_AXIS]));
}
#endif

#endif

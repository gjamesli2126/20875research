/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef AUX_FN
#define AUX_FN
#include<fstream>
#include"Point.h"
void ReadPoints(std::ifstream& input, long int startIndex, long int endIndex, Point* points);
bool ReadPointsInBatch(std::ifstream& input, Point* points, long int totalNumPoints, int& batchSize);
void WritePoints(std::ofstream& output, int procRank, void* inputSet, long int start, long int end);

#endif

/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef BBT_kenrel_hpp
#define BBT_kenrel_hpp

#include "functions.hpp"

extern neighbor *nearest_neighbor;

void k_nearest_neighbor_search(node* tree, datapoint* point, int idx);

#endif /* BBT_kenrel_hpp */

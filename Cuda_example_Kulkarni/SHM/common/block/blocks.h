/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef _BLOCKS_H_
#define _BLOCKS_H_

#include "app_blocks.h"
#include "common.h"

/* blocks.h

Common code for managing blocks of points. This code is *not* application specific.

For the purposes of this file, and blocks.c, Block is an opaque type.
The definitions of these types are in app_blocks.h, and are specific to each benchmark.

*/

//structure for managing multiple call sets
class BlockSet
{
public:
	BlockSet();
	~BlockSet();

	Block *block; //opaque type, defined in app_blocks.h
	Block *next_block; //array of next blocks, one per call set
};

//stack of blocks, one per level of the tree
class BlockStack
{
public:
	BlockStack();
	~BlockStack();

	BlockSet* get(int i);

	BlockSet **items;
	unsigned size; //number of levels
};

class BlockProfiler {
public:
	BlockProfiler() {
		for (int i = 0; i < 5; i++) {
			count[i] = 0;
		}
		block_size_sum = 0;
		block_size_cnt = 0;
	}

	~BlockProfiler() {}

	void output();

	void record(int block_size) {
		count[4] += block_size / 4;
		if (block_size % 4 != 0) count[block_size % 4]++;
		block_size_sum += block_size;
		block_size_cnt++;
	}

	void record_single() { count[0]++; }

	void record_leaf_node_exist_rate(float f) { leaf_node_exist_rate = f; }
private:
	uint64_t count[5];
	uint64_t block_size_sum;
	uint64_t block_size_cnt;
	float leaf_node_exist_rate;
};

#endif

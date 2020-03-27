#ifndef _APP_BLOCKS_H_
#define _APP_BLOCKS_H_

/* app_blocks.h

This header defines the types for the structs that contain the per-point
information, both within the blocked recursive code (Block) and in the
intermediary methods (IntermediaryState).

NOTE: This information is application-specific, and should be defined separately for each benchmark.

This header also provides the signatures for the code needed to construct and tear down arrays of blocks

*/

#include "common.h"

#include "shared/Ray.h"
#include "shared/IntersectInfo.h"


// structure for managing point-specific data that is used during recursive
// traversal
class Block {
public:
	Block();
	~Block();

	void add(Ray &ray, IntersectInfo &info);

	void recycle() { size = 0; }
	bool is_full() { return size == max_block; }
	bool is_empty() { return size == 0; }

	static void set_max_block(int size);
	static int get_max_block() { return max_block; }

	Ray **ray;
	IntersectInfo **info;

	int size; //number of valid points in the block
	static int max_block;
};

const int num_call_sets = 1;

#endif

#ifndef _INTERSTATE_H_
#define _INTERSTATE_H_

#include "app_interstate.h"

/* blocks.h

Common code for managing intermediary state. This code is *not* application specific.

For the purposes of this file, IntermediaryState is an opaque type.
The definitions of these types are in app_interstate.h, and are specific to each benchmark.

*/
class Block;

//structure for managing intermediary state block
class IntermediaryBlock {
public:
	IntermediaryBlock();
	~IntermediaryBlock();

	IntermediaryState* next();
	IntermediaryState* get(int pos);
	void reset();

	static int max_block;
	Block *block;
private:

	int pos;
	IntermediaryState* data; //opaque type, defined in app_blocks.h
};


#endif

/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <assert.h>

#include "harness.h"
#include "app_interstate.h"
#include "interstate.h"

/*

NOTE: the behavior of this code depends on which version of app_interstate.h is included.

*/
int IntermediaryBlock::max_block = 128;

//Create an array of IntermediaryState
IntermediaryBlock::IntermediaryBlock() {
	pos = 0;
	data = new IntermediaryState[max_block];
}

IntermediaryBlock::~IntermediaryBlock() {
	delete [] data;
}

IntermediaryState * IntermediaryBlock::next() {
	assert(pos < max_block);
	return &(data[pos++]);
}

IntermediaryState * IntermediaryBlock::get(int p) {
	assert(p < pos);
	return &(data[p]);
}

void IntermediaryBlock::reset() {
	pos = 0;
}


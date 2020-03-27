/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <cassert>
class Vec ;
class Point ;
class Node ;

struct _IntermediaryState {
	Point* p;
	Vec a_prev;
};

class _IntermediaryBlock {
public:
	_IntermediaryBlock();
	~_IntermediaryBlock();

	_IntermediaryState* next();
	_IntermediaryState* get(int pos);
	void reset();

	_Block *block;
	static int max_block;
private:
	int pos;
	_IntermediaryState* data;
};

int _IntermediaryBlock::max_block = 128;

_IntermediaryBlock::_IntermediaryBlock() {
	pos = 0;
	data = new _IntermediaryState[max_block];
}

_IntermediaryBlock::~_IntermediaryBlock() {
	delete [] data;
}

_IntermediaryState* _IntermediaryBlock::next() {
	assert(pos < max_block);
	return &(data[pos++]);
}

_IntermediaryState* _IntermediaryBlock::get(int p) {
	assert(p < pos);
	return &(data[p]);
}

void _IntermediaryBlock::reset() {
	pos = 0;
}

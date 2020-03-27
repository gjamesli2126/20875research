/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "blocks.h"
#include "common.h"

int Block::max_block;
Node *Block::root;

void Block::set_max_block(int size) {
	if (size <= 0) {
		cout << "block size " << size << endl;
		exit(1);
	}
	max_block = size;
}

Block::Block() {
	points = new Point*[max_block];
	size = 0;
}

Block::~Block() {
	//delete [] points;
}

void Block::add(Point* point) {
	assert(size < max_block);
	points[size++] = point;
}

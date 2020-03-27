
#include "blocks.h"
#include "common.h"

int Block::max_block;

void Block::set_max_block(int size) {
	if (size <= 0) {
		cout << "block size " << size << endl;
		exit(1);
	}
	max_block = size;
}

Block::Block() {
	assert(max_block > 0);
	ray = new Ray*[max_block];
	info = new IntersectInfo*[max_block];
	size = 0;
}

Block::~Block() {
	delete [] ray;
	delete [] info;
}

void Block::add(Ray &ray, IntersectInfo &info) {
	assert(size < max_block);
	this->ray[size] = &ray;
	this->info[size] = &info;
	size++;
}


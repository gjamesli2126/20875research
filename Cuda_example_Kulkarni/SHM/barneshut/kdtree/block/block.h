/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <vector>
#include <cassert>
using namespace std;

class Point ;
class Node ;
class _Point {
public:
	Point* point;
	_Point() {}
	_Point(Point* point);
};

_Point::_Point(Point* point) {
	this->point = point;
}

class _Block {
public:
	_Block();
	~_Block();

	void add(Point* point);
	_Point& get(int i) { return points[i]; }
	void recycle() { size = 0; }
	bool is_full() { return size == max_block; }
	bool is_empty() { return size == 0; }

	_Point* points;
	int size;

	static Node* root_node;
	static int max_block;
};

class _BlockSet
{
public:
	_BlockSet() {}
	~_BlockSet() {}

	_Block *block;
	_Block nextBlock0;
};

class _BlockStack
{
public:
	_BlockStack() {}
	~_BlockStack();

	_BlockSet* get(int i);

	vector<_BlockSet *> items;
};

int _Block::max_block = 128;
Node* _Block::root_node;

_Block::_Block() {
	points = new _Point[max_block];
	size = 0;
}

_Block::~_Block() {
	delete [] points;
}

void _Block::add(Point* point) {
	assert(size < max_block);
	points[size].point = point;
	size++;
}


_BlockStack::~_BlockStack() {
	for (int i = 0; i < items.size(); i++) {
		delete items[i];
	}
}

_BlockSet* _BlockStack::get(int i) {
	while (i >= items.size()) {
		items.push_back(new _BlockSet());
	}
	return items[i];
}

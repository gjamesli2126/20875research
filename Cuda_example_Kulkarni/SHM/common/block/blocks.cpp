/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include "blocks.h"

#include "common.h"
#include "harness.h"

/*

NOTE: the behavior of this code depends on which version of app_blocks.h is included.

 */

BlockSet::BlockSet() {
	assert(num_call_sets > 0);
	next_block = new Block[num_call_sets];
}

BlockSet::~BlockSet() {
	delete [] next_block;
}

const int max_depth = 64;

BlockStack::BlockStack() {
	items = new BlockSet*[max_depth];	// assume max depth = 64
	size = 0;
}

BlockStack::~BlockStack() {
	for (int i = 0; i < size; i++) {
		delete items[i];
	}
	delete [] items;
}

BlockSet* BlockStack::get(int i) {
	assert(i < max_depth);
	while (i >= size) {
		items[size++] = new BlockSet();
	}
	return items[i];
}

void BlockProfiler::output() {
	int weights[5] = {1, 1, 2, 3, 4};

	uint64_t work_sum = 0;
	for (int i = 0; i < 5; i++) {
		work_sum += count[i] * weights[i];
	}
	float work_ratio[5];
	for (int i = 0; i < 5; i++) {
		work_ratio[i] = static_cast<float>(count[i] * weights[i]) / work_sum;
		printf("work %d: %llu %.4f\n", i, count[i] * weights[i], work_ratio[i]);
	}
	printf("work sum: %llu\n", work_sum);
	float block_size_avg = (float) block_size_sum / block_size_cnt;
	printf("block size avg: %.4f\n", block_size_avg);
	printf("block size cnt: %llu\n", block_size_cnt);
	printf("leaf node exist rate: %.4f\n", leaf_node_exist_rate);

	FILE *fp = fopen("blockprofile.csv", "a");

	fprintf(fp, "%s,%s,%d,%d,%d, ",
			Harness::get_benchmark().c_str(), Harness::get_appargs().c_str(),
			Harness::get_sort_flag(), Harness::get_block_size(), Harness::get_splice_depth());
	for (int i = 0; i < 5; i++) {
		fprintf(fp, "%llu,", count[i] * weights[i]);
	}
	fprintf(fp, "%llu,", work_sum);
	for (int i = 0; i < 5; i++) {
		fprintf(fp, "%.4f,", work_ratio[i]);
	}

	fprintf(fp, "%llu, %.4f,", block_size_cnt, block_size_avg);
	fprintf(fp, "%.4f,", leaf_node_exist_rate);
	fprintf(fp, "\n");
	fclose(fp);
}




/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef PROFILERS_H_
#define PROFILERS_H_

#include "harness.h"

class CountProfiler {
public:
	CountProfiler(int n) {
		num = n;
		cnt = new uint64_t[n];
		for (int i = 0; i < n; i++) {
			cnt[i] = 0;
		}
	}

	~CountProfiler() {
		delete [] cnt;
	}

	void record(int i, int n) {
		assert(i < num);
		cnt[i] += n;
	}
	void output() {
		cout << "count profile" << endl;
		for (int i = 0; i < num; i++) {
			//printf("%d %llu\n", i, cnt[i]);
			printf("%llu\n", cnt[i]);
		}

		FILE *fp = fopen("countprofile.csv", "a");
		fprintf(fp, "%s,%s,%d,%d,%d, ",
				Harness::get_benchmark().c_str(), Harness::get_appargs().c_str(),
				Harness::get_sort_flag(), Harness::get_block_size(), Harness::get_splice_depth());
		for (int i = 0; i < num; i++) {
			fprintf(fp, "%llu,", cnt[i]);
		}
		for (int i = 1; i < num; i++) {
			float ratio = (float)cnt[i] / cnt[0];
			fprintf(fp, "%.4f,", ratio);
		}
		fprintf(fp, "\n");
		fclose(fp);
	}

	uint64_t *cnt;
	int num;
};

#endif /* TREESHAPE_H_ */

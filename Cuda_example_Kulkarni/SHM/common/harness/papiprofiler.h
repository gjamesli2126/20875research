/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef PAPIPROFILER_H_
#define PAPIPROFILER_H_

#include "common.h"

class PapiProfiler {
public:
	PapiProfiler();
	~PapiProfiler();

	static void set_arg(int a) { arg = a; }
	void start();
	void stop();
	void start_fine();
	void stop_fine();
	void output(int drop_runs, int actual_runs, float runtimes_avg0, float avgAutotuneBlock, float avgAutotuneDepth);

private:
	static const int max_events = 4;

	static int arg;
	int eventset;
	int num_events;
	vector<long long> values1[max_events];
	vector<long long> values2[max_events];
	long long fine_values1[max_events];	// values within start_fine and stop_fine
	long long fine_values2[max_events];	// all other values, entire run is counted here when start_fine is not used
};

#endif /* PAPIPROFILER_H_ */

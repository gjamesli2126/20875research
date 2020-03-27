/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <papi.h>

#include "harness.h"
#include "papiprofiler.h"

#define call_expect(ret, expected) \
		if (ret != expected) { \
			printf("PAPI error file:%s line:%d ret:%d expected:%d\n", __FILE__, __LINE__, ret, expected); \
			exit(1); \
		}

#define call(ret) call_expect(ret, PAPI_OK);

int PapiProfiler::arg;

PapiProfiler::PapiProfiler() {
	arg = 0;
}

PapiProfiler::~PapiProfiler() {
	// TODO Auto-generated destructor stub
}

void query_events() {
	call(PAPI_query_event(PAPI_TOT_CYC));
	call(PAPI_query_event(PAPI_TOT_INS));
	call(PAPI_query_event(PAPI_L3_DCM)); // not available on Yeti
	call(PAPI_query_event(PAPI_L3_DCA));
	//call(PAPI_query_event(PAPI_VEC_INS));
	//call(PAPI_query_event(PAPI_INT_INS)); // not available on Yeti
	//call(PAPI_query_event(PAPI_FP_INS));

	int num_hwcntrs = PAPI_num_counters();
	printf("This system has %d available counters\n", num_hwcntrs);
}

void PapiProfiler::start() {
	if (arg != 1 && arg !=2 && arg != 10) {
		cout << "papi arg not set" << endl;
		exit(1);
	}
#ifdef PAPI_FINE
	if (arg != 10) {
		cout << "built with PAPI_FINE, arg should be 10" << endl;
		exit(1);
	}
#endif
	call_expect(PAPI_library_init( PAPI_VER_CURRENT ), PAPI_VER_CURRENT);
	call(PAPI_register_thread());

	eventset = PAPI_NULL;
	if (arg == 10) {
		num_events = 2;
		int events[max_events] = { PAPI_TOT_CYC, PAPI_TOT_INS };
		call(PAPI_create_eventset(&eventset));
		call(PAPI_add_events(eventset, events, num_events));
	} else {
		num_events = 4;
		//int events[num_events] = { PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_L1_DCM, PAPI_L1_DCA, PAPI_L2_DCM, PAPI_L2_DCA };
		int events1[max_events] = { PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_L3_TCM, PAPI_L3_TCA };
		//int events2[max_events] = { PAPI_L2_DCM, PAPI_L2_DCA, PAPI_VEC_INS, PAPI_FP_INS };
		//int events1[max_events] = { PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_L1_DCM };
		int events2[max_events] = { PAPI_L1_DCA, PAPI_L2_DCM, PAPI_L2_DCA };
		call(PAPI_create_eventset(&eventset));
		call(PAPI_add_events(eventset, arg == 1 ? events1 : events2, num_events));
	}
	for (int i = 0; i < num_events; i++) {
		fine_values1[i] = 0;
		fine_values2[i] = 0;
	}

	call(PAPI_start(eventset));
}

void PapiProfiler::stop() {
	long long results[max_events];
	call(PAPI_stop(eventset, results));
	for (int i = 0; i < num_events; i++) {
		fine_values2[i] += results[i];
	}

	cout << "papi1: ";
	for (int i = 0; i < num_events; i++) {
		cout << fine_values1[i] << " ";
		values1[i].push_back(fine_values1[i]);
	}
	cout << endl;
	cout << "papi2: ";
	for (int i = 0; i < num_events; i++) {
		cout << fine_values2[i] << " ";
		values2[i].push_back(fine_values2[i]);
	}
	cout << endl;
	printf("Average L3 miss rate:%f Avg CPI:%f\n",fine_values2[2]/(double)fine_values2[3],fine_values2[0]/(double)fine_values2[1]);
}

void PapiProfiler::start_fine() {
	long long results[max_events];
	call(PAPI_accum(eventset, fine_values2));
}

void PapiProfiler::stop_fine() {
	long long results[max_events];
	call(PAPI_accum(eventset, fine_values1));
}

float get_mean_min_max(int start, int end, const vector<long long>& runtimes, long long *min, long long *max) {
	long long runtimes_sum = 0;
	long long runtimes_max = 0;
	long long runtimes_min = 1LL << 60;
	float runtimes_avg;

	for (int i = start; i < end; i++) {
		runtimes_sum += runtimes[i];
		runtimes_min = runtimes_min < runtimes[i] ? runtimes_min : runtimes[i];
		runtimes_max = runtimes_max > runtimes[i] ? runtimes_max : runtimes[i];
	}
	runtimes_avg = (float) runtimes_sum / (end - start);
	if (min != NULL) *min = runtimes_min;
	if (max != NULL) *max = runtimes_max;
	return runtimes_avg;
}

float get_stability_stdev(int start, int end, const vector<long long>& runtimes, float mean, float *stdev) {
	float runtimes_variance_sum = 0;
	float runtimes_variance;
	float runtimes_stdev;
	float runtimes_stability;
	for (int i = start; i < end; i++) {
		runtimes_variance_sum = (runtimes[i] - mean) * (runtimes[i] - mean);
	}
	if (end - start > 1) runtimes_variance = runtimes_variance_sum / (end - start - 1);
	else runtimes_variance_sum = 0;
	runtimes_stdev = sqrt(runtimes_variance);
	runtimes_stability = runtimes_stdev * 100 / mean;
	if (stdev) *stdev = runtimes_stdev;
	return runtimes_stability;
}

void output_values(FILE *fp, int drop_runs, int actual_runs, const vector<long long>* values, int num) {
	for (int i = 0; i < num; i++) {
	 	long long min, max;
		float stdev;
		float avg = get_mean_min_max(drop_runs, actual_runs, values[i], &min, &max);
		float stability = get_stability_stdev(drop_runs, actual_runs, values[i], avg, &stdev);
		fprintf(fp, "%.2f, %lld, %lld, %.2f, %.2f, ",
				avg, min, max,stdev, stability);
		printf("[papi %d:%.2f][min:%lld][max:%lld][stability:%.2f]\n",
				i, avg, min, max, stability);
	}
}

void output_prefix(FILE *fp, int drop_runs, int actual_runs, float runtimes_avg0, float avgAutotuneBlock, float avgAutotuneDepth) {
	fprintf(fp, "%s,%s,%d,%d,%d,%d,%d, ",
			Harness::get_benchmark().c_str(), Harness::get_appargs().c_str(),
			Harness::get_sort_flag(), actual_runs, drop_runs, Harness::get_block_size(), Harness::get_splice_depth());
	fprintf(fp, "%.2f,%.2f,%.2f, ", runtimes_avg0, avgAutotuneBlock, avgAutotuneDepth);
}

void PapiProfiler::output(int drop_runs, int actual_runs, float runtimes_avg0, float avgAutotuneBlock, float avgAutotuneDepth) {
	bool print_header = false;
	FILE *fp = fopen("papi.csv", "r");
	if (fp == NULL) {
		print_header = true;
	} else {
		fclose(fp);
	}

	fp = fopen("papi.csv", "a");
	assert(fp != NULL);
	if (print_header) {
		fprintf(fp, "benchmark, args, sort_flag, actualruns, dropruns, block_size, splice_depth, avg, min, max, stdev, stability\n");
	}

	if (arg == 10) {
		output_prefix(fp, drop_runs, actual_runs, runtimes_avg0, avgAutotuneBlock, avgAutotuneDepth);
		output_values(fp, drop_runs, actual_runs, values1, num_events);
		output_values(fp, drop_runs, actual_runs, values2, num_events);
		fprintf(fp, "\n");
	} else {
		if (arg == 1) {
			output_prefix(fp, drop_runs, actual_runs, runtimes_avg0, avgAutotuneBlock, avgAutotuneDepth);
		}
		output_values(fp, drop_runs, actual_runs, values2, num_events);
		if (arg == 2) {
			fprintf(fp, "\n");
		}
	}

	fclose(fp);
}



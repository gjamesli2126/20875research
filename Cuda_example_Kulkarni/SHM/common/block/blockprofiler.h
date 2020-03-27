/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef BLOCKPROFILER_H_
#define BLOCKPROFILER_H_

typedef vector<int> CountVec;

class CountStack
{
public:
	CountStack() {}
	~CountStack() {
		for (int i = 0; i < items.size(); i++) {
			delete items[i];
		}
	}

	CountVec* get(int i) {
		while (i >= items.size()) {
			items.push_back(new vector<int>());
		}
		return items[i];
	}

	int size() { return items.size(); }

	vector<CountVec *> items;
};

class BlockProfiler {
public:
	BlockProfiler(int width = 4) {
		this->width = width;
		count = new uint64_t[width + 1];
		for (int i = 0; i < width + 1; i++) {
			count[i] = 0;
		}
		block_size_sum = 0;
		block_size_cnt = 0;
	}

	~BlockProfiler() {
		delete [] count;
	}

	void output() {
		printf("--- Block Profile ---\n");
		uint64_t work_sum = 0;
		for (int i = 0; i < width + 1; i++) {
			int weight = i == 0 ? 1 : i;
			work_sum += count[i] * weight;
		}
		float *work_ratio = new float[width + 1];
		for (int i = 0; i < width + 1; i++) {
			int weight = i == 0 ? 1 : i;
			work_ratio[i] = static_cast<float>(count[i] * weight) / work_sum;
			printf("work %d: %llu %.4f\n", i, count[i] * weight, work_ratio[i]);
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
		for (int i = 0; i < width + 1; i++) {
			int weight = i == 0 ? 1 : i;
			fprintf(fp, "%llu,", count[i] * weight);
		}
		fprintf(fp, "%llu,", work_sum);
		for (int i = 0; i < width + 1; i++) {
			fprintf(fp, "%.4f,", work_ratio[i]);
		}

		fprintf(fp, "%llu, %.4f,", block_size_cnt, block_size_avg);
		fprintf(fp, "%.4f,", leaf_node_exist_rate);
		fprintf(fp, "\n");
		fclose(fp);

		delete [] work_ratio;

		outputParameter();
	}

	void outputParameter() {
		printf("--- Parameter Profile ---\n");

		uint64_t totalWork = 0;
		int totalCalls = 0;
		vector<float> means;
		vector<int> mins;
		vector<int> maxs;
		vector<int> calls;
		vector<float> stabilities;
		for (int i = 0; i < countStack.size(); i++) {
			CountVec *cvec = countStack.get(i);
			int size = cvec->size();
			uint64_t sum = 0;
			int max = 0;
			int min = 1 << 30;
			for (int j = 0; j < size; j++) {
				int val = cvec->at(j);
				sum += val;
				min = min < val ? min : val;
				max = max > val ? max : val;
			}
			totalWork += sum;
			float mean = (float) sum / size;
			sum = 0;
			for (int j = 0; j < size; j++) {
				sum += (cvec->at(j) - mean) * (cvec->at(j) - mean);
			}
			float variance = sum / (size - 1);
			float stdev = sqrt(variance);
			float stability = stdev * 100 / mean;
			means.push_back(mean);
			mins.push_back(min);
			maxs.push_back(max);
			stabilities.push_back(stability);
			calls.push_back(size);
			totalCalls += size;
			printf("%d \t %d \t %.1f \t %d \t %d \t %.2f\n", i, size, mean, min, max, stability);
		}

		FILE *fp = fopen("parameter.csv", "a");

		fprintf(fp, "%s,%s,%d,%d,%d, ",
				Harness::get_benchmark().c_str(), Harness::get_appargs().c_str(),
				Harness::get_sort_flag(), Harness::get_block_size(), Harness::get_splice_depth());
		float totalMean = (float) totalWork / totalCalls;
		printf("totalMean: %.1f\n", totalMean);
		uint64_t nonBlockedWork = count[0];
		totalWork += nonBlockedWork;	// exclude nonBlockedWork in totalMean
		fprintf(fp, "%llu,", totalWork);
		fprintf(fp, "%.1f,", totalMean);
		float nonBlockedWorkRatio = (float) nonBlockedWork / totalWork;
		fprintf(fp, "%.4f,mean,", nonBlockedWorkRatio);
		for (int i = 0; i < means.size(); i++) {
			fprintf(fp, "%.1f,", means[i]);
		}
		fprintf(fp, "\n,,,,,,,,min, ");
		for (int i = 0; i < mins.size(); i++) {
			fprintf(fp, "%d,", mins[i]);
		}
		fprintf(fp, "\n,,,,,,,,max, ");
		for (int i = 0; i < maxs.size(); i++) {
			fprintf(fp, "%d,", maxs[i]);
		}
		fprintf(fp, "\n,,,,,,,,stability, ");
		for (int i = 0; i < stabilities.size(); i++) {
			fprintf(fp, "%.2f,", stabilities[i]);
		}
		fprintf(fp, "\n,,,,,,,,calls, ");
		for (int i = 0; i < calls.size(); i++) {
			fprintf(fp, "%d,", calls[i]);
		}
		fprintf(fp, "\n");
		fclose(fp);
	}

	void record(int block_size) {
		count[width] += block_size / width;
		if (block_size % width != 0) count[block_size % width]++;
		block_size_sum += block_size;
		block_size_cnt++;
	}

	void record(int size, int depth) {
		CountVec* cvec = countStack.get(depth);
		cvec->push_back(size);

		record(size);
	}

	void record_single() { count[0]++; }

	void record_leaf_node_exist_rate(float f) { leaf_node_exist_rate = f; }
private:
	CountStack countStack;
	int width;
	uint64_t *count;
	uint64_t block_size_sum;
	uint64_t block_size_cnt;
	float leaf_node_exist_rate;
};

#endif /* BLOCKPROFILER_H_ */

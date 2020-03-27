/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef PARALLELISMPROFILER_H_
#define PARALLELISMPROFILER_H_


class ParallelismProfiler {
public:
	ParallelismProfiler(int w = Harness::get_vectorWidth(), int costT = Harness::get_truncateCost(), int costR = Harness::get_recurseCost())
	: width(w), costRecurse(costR), costTruncate(costT) {
		timeSerial = 0;
		timeNonBlocked = 0;
		timeBlockedSingleType = 0;
		timeBlockedBest = 0;
		timeBlockedActual = 0;
		timeBlockedWorst = 0;
		countTruncateNonBlocked = 0;
		countRecurseNonBlocked = 0;
		countTruncateBlocked = 0;
		countRecurseBlocked = 0;
		blockClear();
	}

	~ParallelismProfiler() {
	}

	void output() {
		printf("--- Parallelism Profile ---\n");
		cout << "timeSerial: " << timeSerial << endl;
		cout << "timeNonBlocked: " << timeNonBlocked << endl;
		cout << "timeBlockedSingleType: " << timeBlockedSingleType << endl;
		cout << "timeBlockedBest: " << timeBlockedBest << endl;
		cout << "timeBlockedActual: " << timeBlockedActual << endl;
		cout << "timeBlockedWorst: " << timeBlockedWorst << endl;
		cout << "countTruncateNonBlocked: " << countTruncateNonBlocked << endl;
		cout << "countRecurseNonBlocked: " << countRecurseNonBlocked << endl;
		cout << "countTruncateBlocked: " << countTruncateBlocked << endl;
		cout << "countRecurseBlocked: " << countRecurseBlocked << endl;

		FILE *fp = fopen("vectorprofile.csv", "a");

		fprintf(fp, "%s,%s,%d,%d,%d,%d,%d,%d, ",
				Harness::get_benchmark().c_str(), Harness::get_appargs().c_str(),
				Harness::get_sort_flag(), Harness::get_block_size(), Harness::get_splice_depth(),
				Harness::get_vectorWidth(), Harness::get_truncateCost(), Harness::get_recurseCost());
		fprintf(fp, "%lld,%lld,%lld,%lld,%lld,%lld, %lld,%lld,%lld,%lld,",
				timeSerial, timeNonBlocked, timeBlockedSingleType, timeBlockedBest, timeBlockedActual, timeBlockedWorst,
				countTruncateNonBlocked, countRecurseNonBlocked, countTruncateBlocked, countRecurseBlocked);
		fprintf(fp, "\n");
		fclose(fp);
	}

	void recordNonBlockedTruncate() {
		timeNonBlocked += costTruncate;
		timeSerial += costTruncate;
		countTruncateNonBlocked++;
	}

	void recordNonBlockedRecurse() {
		timeNonBlocked += costRecurse;
		timeSerial += costRecurse;
		countRecurseNonBlocked++;
	}

	void recordTruncate() {
		timeSerial += costTruncate;
		countTruncate++;
		countTruncateBlocked++;
		workVector.push_back(TRUNCATE);
	}

	void recordRecurse() {
		timeSerial += costRecurse;
		countRecurse++;
		countRecurseBlocked++;
		workVector.push_back(RECURSE);
	}

	void blockEnd() {
		/*
We have two types of work truncate (T) and recurse (R).
We want to gather three numbers:
1) best case: process all T first, then all R
2) actual case: process in order they arise
3) worst case: interleave T and R for worst performance

If we have 8 Ts and 8 Rs on a 4-way SIMD unit
1) TTTT, TTTT, RRRR, RRRR = 4 cycles
3) TTxx, xxRR, TTxx, xxRR, TTxx, xxRR, TTxx, xxRR = 8 cycles

For the worst case:

If both T & R are greater than (T + R)/S, then every SIMD group will take T + R cycles.

So with my example of 17 Ts and 27 Rs, and S=8, it would take ceil((17+27)/8) * 2 = 12 cycles, since we can put a T in each group of 8 work.

If T or R is less than (T + R)/S, then we would fall back to my proposed worst case, putting one T in each group until T runs out.
So with 7 Ts and 127 Rs, and S=8,
Txxxxxxx
xRRRRRRR consumes all 7 Ts, and 49 Rs in 14 cycles.
Then the remaining 78 Rs take 10 more cycles.

This would generalize to arbitrary cycle costs for T and R.
		 */
		if (costTruncate == costRecurse) {
			// single type only makes sense when T and R have same cost
			int countSingleType = countRecurse + countTruncate;
			int remSingleType = countSingleType % width;
			timeBlockedSingleType += (countSingleType / width);
			if (remSingleType > 0) timeBlockedSingleType++;
		}
		int remTruncate = countTruncate % width;
		int remRecurse = countRecurse % width;
		timeBlockedBest += (countTruncate / width) * costTruncate;
		if (remTruncate > 0) timeBlockedBest += costTruncate;
		timeBlockedBest += (countRecurse / width) * costRecurse;
		if (remRecurse > 0) timeBlockedBest += costRecurse;

		for (int i = 0; i < workVector.size(); i += width) {
			int existTruncate = 0;
			int existRecurse = 0;
			int limit = ((i + width) < workVector.size()) ? (i + width) : workVector.size();
			for (int j = i; j < limit; j++) {
				//cout << workVector[j] << " ";
				if (workVector[j] == TRUNCATE) existTruncate = costTruncate;
				else existRecurse = costRecurse;
			}

			timeBlockedActual += (existTruncate + existRecurse);
			//cout << endl << timeBlockedActual << " " << existTruncate << " " << existRecurse << endl;
		}

		int count = countTruncate + countRecurse;
		if (countTruncate < countRecurse) {
			if (countTruncate >= count / width) {
				int remCount = count % width;
				timeBlockedWorst += (count / width) * (costTruncate + costRecurse);
				if (remCount > 0) timeBlockedWorst += (costTruncate + costRecurse);
			} else {
				timeBlockedWorst += (countTruncate * (costTruncate + costRecurse));
				countRecurse -= (width - 1) * countTruncate;
				int remRecurse = countRecurse % width;
				timeBlockedWorst += (countRecurse / width) * costRecurse;
				if (remRecurse > 0) timeBlockedWorst += costRecurse;
			}
		} else {
			if (countRecurse >= count / width) {
				int remCount = count % width;
				timeBlockedWorst += (count / width) * (costTruncate + costRecurse);
				if (remCount > 0) timeBlockedWorst += (costTruncate + costRecurse);
			} else {
				timeBlockedWorst += (countRecurse * (costTruncate + costRecurse));
				countTruncate -= (width - 1) * countRecurse;
				int remRecurse = countTruncate % width;
				timeBlockedWorst += (countTruncate / width) * costTruncate;
				if (remRecurse > 0) timeBlockedWorst += costTruncate;
			}
		}

		blockClear();
	}

	void blockClear() {
		countTruncate = 0;
		countRecurse = 0;
		workVector.clear();
	}

	static void test() {
		ParallelismProfiler p(8);
		for (int i = 0; i < 10; i++) p.recordTruncate();
		for (int i = 0; i < 27; i++) p.recordRecurse();
		for (int i = 0; i < 7; i++) p.recordTruncate();
		p.blockEnd();
		p.output();

		cout << endl << endl;

		ParallelismProfiler p2(8);
		for (int i = 0; i < 7; i++) p2.recordTruncate();
		for (int i = 0; i < 127; i++) p2.recordRecurse();
		p2.blockEnd();
		p2.output();
	}

private:
	enum {
		TRUNCATE, RECURSE
	};

	const int width;
	const int costTruncate, costRecurse;

	vector<int> workVector;

	// stats per block, reset at blockEnd()
	int countTruncate, countRecurse;

	// global stats
	uint64_t countTruncateBlocked, countRecurseBlocked;
	uint64_t countTruncateNonBlocked, countRecurseNonBlocked;

	uint64_t timeSerial, timeNonBlocked;
	uint64_t timeBlockedSingleType;
	uint64_t timeBlockedBest, timeBlockedActual, timeBlockedWorst;
};

#endif /* PARALLELISMPROFILER_H_ */

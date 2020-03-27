/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef AUTOTUNER_H_
#define AUTOTUNER_H_

class _Autotuner {
public:
	_Autotuner(int num);
	~_Autotuner();

	int **tune();
	void tuneSetup(int num);
	void tuneFinished();
	bool isSampled(int i);

	const static int maxTuneCnt = 5;
	int *sampleSizes;
	int tuneIndexesCnt;

	// Block specific
	void profileWorkDone(int work);
	void tuneEntryBlock();
	void tuneExitBlock(int i);

private:
	void setupSample(int idx, int size);
	void setSampled(int i);
	void nextSample();

	const static int startBlockSize = 8;

	int **tuneIndexes;

	int samplePoint;
	int numPoints;
	bool *sampled;

	// Block specific
	void tuneSetupBlock();
	void tuneCleanupBlock();

	int workDone;
	int maxBlockSize;
	float *normalizedTimes;
	int *blockSizes;
	int blockSizesCnt;
	timeval timeStart, timeEnd;
};

_Autotuner::_Autotuner(int num) {
	srand(time(NULL));
	tuneSetup(num);
	tuneSetupBlock();
}

_Autotuner::~_Autotuner() {
	tuneCleanupBlock();
	for (int i = 0; i < maxTuneCnt; i++) {
		delete [] tuneIndexes[i];
	}
	delete [] tuneIndexes;
	delete [] sampleSizes;
	delete [] sampled;
}

int **_Autotuner::tune() {
	tuneIndexesCnt = maxTuneCnt * blockSizesCnt;
	tuneIndexes = new int*[tuneIndexesCnt];
	sampleSizes = new int[tuneIndexesCnt];
	for (int i = 0; i < maxTuneCnt; i++) {
		for (int j = 0; j < blockSizesCnt; j++) {
			int size = blockSizes[j];
			setupSample(i * blockSizesCnt + j, size);
		}
	}
	return tuneIndexes;
}

void _Autotuner::tuneFinished() {

	float minTime = FLT_MAX;
	int bestIndex = 0;
	for (int j = 0; j < blockSizesCnt; j++) {
		//cout << blockSizes[j] << "\t" << normalizedTimes[j] << endl;
		if (normalizedTimes[j] < minTime) {
			minTime = normalizedTimes[j];
			bestIndex = j;
		}
	}
	int block = blockSizes[bestIndex];
	int depth = 0;
	Block::max_block = block;
	_IntermediaryBlock::max_block = block;

	printf("autotune params [block:%d][depth:%d]\n", block, depth);
	Harness::recordAutotuneParams(block, depth);
}

bool _Autotuner::isSampled(int i) {
	return sampled[i];
}

void _Autotuner::setSampled(int i) {
	sampled[i] = true;
}

void _Autotuner::setupSample(int idx, int size) {
	sampleSizes[idx] = size;
	tuneIndexes[idx] = new int[size];
	samplePoint = rand() % numPoints;
	for (int i = 0; i < size; i++) {
		nextSample();
		tuneIndexes[idx][i] = samplePoint;
	}
}

void _Autotuner::nextSample() {
	samplePoint++;
	if (samplePoint == numPoints) {
		samplePoint = 0;
	}
	while (isSampled(samplePoint)) {
		samplePoint++;
		if (samplePoint == numPoints) {
			samplePoint = 0;
		}
	}
	setSampled(samplePoint);
}

void _Autotuner::tuneSetup(int num) {
	numPoints = num;
	sampled = new bool[numPoints];
	memset(sampled, 0, sizeof(bool) * num);
}

// Block specific


void _Autotuner::tuneSetupBlock() {
	workDone = 0;
	int tuneFraction = Harness::get_autotuneFraction();
	int limit = numPoints / (tuneFraction * maxTuneCnt * 2);
	maxBlockSize = startBlockSize;
	int cnt = 1;
	while (maxBlockSize * 2 < limit) {
		maxBlockSize *= 2;
		cnt++;
	}
	blockSizesCnt = cnt;
	normalizedTimes = new float[blockSizesCnt];
	memset(normalizedTimes, 0, sizeof(float) * blockSizesCnt);
	blockSizes = new int[blockSizesCnt];
	for (int i = 0, size = startBlockSize; i < cnt; i++, size *= 2) {
		blockSizes[i] = size;
	}

	Block::max_block = maxBlockSize;
	_IntermediaryBlock::max_block = maxBlockSize;
}

void _Autotuner::tuneCleanupBlock() {
	delete [] normalizedTimes;
	delete [] blockSizes;
}

void _Autotuner::profileWorkDone(int work) {
	workDone += work;
}

void _Autotuner::tuneEntryBlock() {
	workDone = 0;
	gettimeofday(&timeStart, NULL);
}

void _Autotuner::tuneExitBlock(int i) {
	i = i % blockSizesCnt;
	gettimeofday(&timeEnd, NULL);
	int sec = timeEnd.tv_sec - timeStart.tv_sec;
	int usec = timeEnd.tv_usec - timeStart.tv_usec;
	long elapsed = (sec * 1000000) + usec;
	float normalizedTime = (float) elapsed / workDone;
	//cout << i << "\t" << elapsed << "\t" << workDone << endl;
	normalizedTimes[i] += normalizedTime;
}


#endif /* AUTOTUNER_H_ */

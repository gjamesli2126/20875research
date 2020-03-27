/*
 * autotuner.h
 *
 *  Created on: May 16, 2013
 *      Author: yjo
 */

#ifndef AUTOTUNER_H_
#define AUTOTUNER_H_

#include "common.h"

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



#endif /* AUTOTUNER_H_ */

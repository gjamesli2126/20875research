/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef TIMER_H
#define TIMER_H
#include<stddef.h>
#include <sys/time.h>
#include<stdint.h>
class timer{
	public:
		timer();
		void           Start();
		void           Stop();
		uint64_t		GetTotTime();
		bool IsRunning();
		uint64_t		GetLastExecTime();
	private:
		struct timeval  startTime;
		struct timeval  endTime;
		uint64_t totTime;
		bool running;
		uint64_t lastExTime;
};

#endif

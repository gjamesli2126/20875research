#ifndef TIMER_H
#define TIMER_H
#include<stddef.h>
#include <sys/time.h>
#include<stdint.h>
class Timer{
	public:
		Timer();
		void           Start();
		void           Stop();
		uint64_t		GetTotTime();
		//uint64_t		GetLastExecTime();
	private:
		struct timeval  startTime;
		struct timeval  endTime;
		uint64_t totTime;
		bool running;
		uint64_t lastExTime;
};

#endif

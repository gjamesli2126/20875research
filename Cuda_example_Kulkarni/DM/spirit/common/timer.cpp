#include"timer.h"

Timer::Timer()
{
	totTime = 0.0;
	lastExTime = 0.0;
	running=false;
}


void Timer::Start() 
{
	if(running)
		return;
	gettimeofday(&startTime,NULL);
	running=true;
}


void Timer::Stop()
{
	if(!running)
		return;
	running = false;
	gettimeofday(&endTime,NULL);
	int sec = endTime.tv_sec - startTime.tv_sec;
	int usec = endTime.tv_usec - startTime.tv_usec;
	lastExTime  = (sec * 1000000) + usec;
	totTime += lastExTime;
}

uint64_t Timer::GetTotTime()
{
	return totTime;
}

/*uint64_t Timer::GetLastExecTime()
{
	return lastExTime;
}*/

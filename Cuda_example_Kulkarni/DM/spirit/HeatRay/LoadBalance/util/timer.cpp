#include"timer.h"
#include<assert.h>

timer::timer()
{
	totTime = 0.0;
	lastExTime = 0.0;
	running=false;
}


void timer::Start() 
{
	if(running)
		return;
	gettimeofday(&startTime,NULL);
	running=true;
}


void timer::Stop()
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

double timer::GetTotTime()
{
	return totTime;
}

uint64_t timer::GetLastExecTime()
{
	return lastExTime;
}

double timer::Reset()
{
	totTime = 0;
	running = false;
}

bool timer::IsRunning()
{
	return running;
}

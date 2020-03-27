/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
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
	/*if(running)
		return;*/
	assert(!running);
	gettimeofday(&startTime,NULL);
	running=true;
}


void timer::Stop()
{
	assert(running);
	/*if(!running)
		return;*/
	running = false;
	gettimeofday(&endTime,NULL);
	int sec = endTime.tv_sec - startTime.tv_sec;
	int usec = endTime.tv_usec - startTime.tv_usec;
	lastExTime  = (sec * 1000000) + usec;
	totTime += lastExTime;
}

uint64_t timer::GetTotTime()
{
	assert(!running);
	return totTime;
}

uint64_t timer::GetLastExecTime()
{
	return lastExTime;
}

bool timer::IsRunning()
{
	return running;
}

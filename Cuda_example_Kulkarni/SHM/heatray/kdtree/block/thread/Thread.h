/*

   Filename : Thread.h
   Author   : Cody White
   Version  : 1.1

   Purpose  : This class uses C++ fast delegates in order to implement
              UNIX/Windows threads.  The use of delegates will get around the
              common pitfalls of C++ thread which either require an
              inheritance architecture or the use of static functions
              inside of a class.  This class will allow the user to simply
              execute functions in their classes on seperate threads, no
              casting, static, or inheritance required.  Just instantiate
              and go!

   Change List:

      - 02/04/2009  - Created (Cody White)

      - 02/10/2009  - Added the ThreadMutex class and the ThreadEnd delegate (Cody White)

	  - 07/29/2009  - Added Windows compatability (Cody White)
*/

/*   Example usage of this class

1)
     SomeClass a;
     Thread thread (ThreadStart (&a, &SomeClass::someFunctionInSomeClass));
     thread.start ();

2)
     SomeClass a;
     Thread thread (ThreadStart (&a, &SomeClass::someFunctionInSomeClass), ThreadEnd (&a, &SomeClass::someFunction));
     thread.setParameter ((void *) param);
     thread.start ();

3)
     Thread thread;
     thread.setStartFunction (ThreadStart (this, &SomeClass::someFunctionInthis));
     thread.setEndFunction (ThreadEnd (this, &SomeClass::someFunction));
     thread.start ();

4)
     Thread thread;
     ThreadStart delegate (this, &SomeClaas::someFunctionInthis);
     thread.setStartFunction (delegate);
     thread.start ();

5)
     // Set the priority to 10 in initialization.
     Thread thread (ThreadStart (this, &SomeClass::someFunctionInthis), 10);
     thread.start ();

6)
     ThreadMutex mutex;
     mutex.lock ();
     mutex.unlock ();
     mutex.tryLock ();

*/

#pragma once

#ifndef __THREAD_H__
#define __THREAD_H__

#ifndef FASTDELEGATE_H
#include "FastDelegate.h"
#endif

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
	#define WINDOWS
#endif

#ifdef WINDOWS
	#include <windows.h>
#else
	#include <pthread.h>
	#include <stdio.h>
#endif

#include <iostream>

using namespace fastdelegate;

// Define a usable delegate type to use with this class that can be used outside of the
// scope of this class.  This type has no parameters as the threads will not allow parameters to be passed
// to the thread's function, else a set amount of parameters would need to be passed always.  To set
// variables for the thread to use, make the variables either global or part of the class.  This names the
// function the thread will call to run.
typedef FastDelegate1 <void *> ThreadStart;

// This is the delegate used when a thread has finished.
typedef FastDelegate1 <void *> ThreadEnd;

/* Example use of ThreadStart --

   ThreadStart delegate (this, &ClassName::functionName);

   This code will publish the delegate to work with the instance of the class referenced by
   'this' with the function functionName.

   ThreadStart must point to a function with one void * parameter and a void return type, e.g.
   void ClassName::functionName (void *foo).  
*/

/*******************************************************************************************

                                    Thread Mutex Class

*******************************************************************************************/
class ThreadMutex
{
   public:

      // Default Constructor.**************************************************************
      ThreadMutex (void)
      {
         // Initialize the thread mutex.
		 #ifdef WINDOWS
			m_thread_mutex = CreateMutex (NULL, FALSE, NULL);
			m_initialized = (m_thread_mutex == 0) ? false : true;
		 #else
			m_initialized = (pthread_mutex_init (&m_thread_mutex, NULL ) == 0);
		 #endif
      }

      // Default Destructor.***************************************************************
      ~ThreadMutex (void)
      {
         // Make sure the mutex is unlocked, then destroy it.
         if (m_initialized)
         {
            unlock ();
			#ifdef WINDOWS
				CloseHandle (m_thread_mutex);
			#else
				pthread_mutex_destroy (&m_thread_mutex);
			#endif

            m_initialized = false;
         }
      }

      // Lock the mutex.*******************************************************************
      bool lock (void)
      {
         if (m_initialized)
         {
			#ifdef WINDOWS
				return WaitForSingleObject (m_thread_mutex, INFINITE) != WAIT_FAILED;
			#else
				return (pthread_mutex_lock (&m_thread_mutex) == 0);
			#endif
         }

		 return false;
      }

#ifndef WINDOWS
      // Attempt to get a lock.  If one is not gotten, this function simply returns and continues
      // execution, not blocking the calling thread.
      bool tryLock (void)
      {
         if (m_initialized)
         {
            return (pthread_mutex_trylock (&m_thread_mutex) == 0);
         }
      }
#endif

      // Unlock the mutex.****************************************************************
      bool unlock (void)
      {
         if (m_initialized)
         {
			#ifdef WINDOWS
				return ReleaseMutex (m_thread_mutex) != 0;
			#else
				return (pthread_mutex_unlock (&m_thread_mutex) == 0);
			#endif
         }

		 return false;
      }

#ifdef WINDOWS
	HANDLE *getMutex (void)
	{
		return m_thread_mutex;
	}
#else
	pthread_mutex_t *getMutex (void)
	{
		return &m_thread_mutex;
	}
#endif

   private:

      // Member variables
	  #ifdef WINDOWS
		  HANDLE m_thread_mutex;
	  #else
		  pthread_mutex_t m_thread_mutex; 
	  #endif

		  bool m_initialized;
};

#endif

/*******************************************************************************************

                                    Thread Class

*******************************************************************************************/

class Thread
{
   public:

      // Error code enum.  One of these codes will be returned after a call to
      // create or stop the thread.
      enum ErrorCodes
      {
         SUCCESSFUL              = 1,
         INVALID                 = 2,
         ALREADY_RUNNING         = 3,
         COULD_NOT_CREATE_THREAD = 4,
         NO_DELEGATE_SPECIFIED   = 5,
         THREAD_NOT_RUNNING      = 6,
         THREAD_INVALID          = 7,
         COULD_NOT_CANCEL_THREAD = 8,
         COULD_NOT_SET_PRIORITY  = 9,
         COULD_NOT_SET_POLICY	 = 10,
         UNKNOWN_SCHED_POLICY    = 11
      };
      
      // Scheduling policies.
      enum SchedPolicy
      {
      	FIFO,
      	ROUND_ROBIN,
      	AUTO 			// determined by the OS.
      };

      // Default constructor************************************************************************
      Thread (void)
      {
         m_start_delegate = NULL;
         m_thread_alive = false;
         m_thread = NULL;
		 #ifndef WINDOWS
				  pthread_attr_init (&m_thread_attributes);
				  pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
		 #endif
      }

      // Constructor 1 -- Pass in start delegate to use.********************************************
      Thread (ThreadStart del, const int priority = -1)
      {
         m_start_delegate = NULL;
         m_thread_alive = false;
         m_thread = NULL;
		 #ifndef WINDOWS
			 pthread_attr_init (&m_thread_attributes);
			 pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
		 #endif
         if (del)
         {
            m_start_delegate = del;
         }
         if (priority != -1) // Set the priority if the user has passed one in.
         {
            setPriority (priority);;
         }
      }

      // Constructor 2 -- Pass in start and end delegate to use.*************************************
      Thread (ThreadStart start_del, ThreadEnd end_del, const int priority = -1)
      {
         m_start_delegate = NULL;
         m_thread_alive = false;
         m_thread = NULL;
         #ifndef WINDOWS
			 pthread_attr_init (&m_thread_attributes);
			 pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
		 #endif
         if (start_del)
         {
            m_start_delegate = start_del;
         }
         if (end_del)
         {
            m_end_delegate = end_del;
         }
         if (priority != -1) // Set the priority if the user has passed one in.
         {
            setPriority (priority);
         }
      }

      // Destructor*********************************************************************************
      ~Thread (void)
      {
		  #ifndef WINDOWS
			 pthread_attr_destroy (&m_thread_attributes);
		  #endif
      }

      // Set the current start delegate to the passed in delegate.**********************************
      void setStartFunction (ThreadStart delegate)
      {
         if (delegate)
         {
            m_start_delegate = delegate;
         }
      }

      // Set the current end delegate to the passed in delegate.************************************
      void setEndFunction (ThreadEnd delegate)
      {
         if (delegate)
         {
            m_end_delegate = delegate;
         }
      }

	  // Set the parameter to pass to the thread.***************************************************
	  void setParameter (void *param)
	  {
		  m_param = param;
	  }

	  // Set the end parameter to pass from the thread.*********************************************
	  void setEndParameter (void *param)
	  {
		  m_end_param = param;
	  }

      // Start the thread***************************************************************************
      int start (void)
      {
         int return_value = ALREADY_RUNNING;
		 if (!m_thread_alive) // Do not start the thread unless it is not already running.
         {
            return_value = NO_DELEGATE_SPECIFIED;
            if (m_start_delegate)
            {
               return_value = COULD_NOT_CREATE_THREAD;
			   m_thread_alive = true;
			   return_value = SUCCESSFUL;
			   #ifdef WINDOWS
				   if ((m_thread = CreateThread (NULL, 0, (LPTHREAD_START_ROUTINE) &Thread::run, this, 0, NULL)) != NULL)
				   {
					   m_thread_alive = false;
				   }
			   #else
				   if (pthread_create (&m_thread, &m_thread_attributes, &Thread::run, this) != 0)
				   {
					  m_thread_alive = false;
				   }
			   #endif
            }
         }

         return return_value;
      }

      // Stop the currently running thread.*********************************************************
      int stop (void)
      {
         int return_value = THREAD_NOT_RUNNING;
         if (m_thread_alive)  // Only stop a thread that is running.
         {
            return_value = THREAD_INVALID;
            if (m_thread)
            {
               return_value = COULD_NOT_CANCEL_THREAD;
			   #ifdef WINDOWS
				   if (TerminateThread (m_thread, 0))
				   {
					   m_thread_alive = false;
					   return_value = SUCCESSFUL;
				   }
			   #else
				   if (pthread_cancel (m_thread) == 0)
				   {
					  m_thread_alive = false;
					  return_value = SUCCESSFUL;
				   }
			   #endif
            }
         }

         return return_value;
      }

      // Stop the calling thread until this thread has finished execution.**************************
      void join (void)
      {
         if (m_thread_alive)
         {
			 #ifdef WINDOWS
				WaitForSingleObject (m_thread, INFINITE);
			 #else
				pthread_join (m_thread, NULL);
			 #endif
         }
      }

      // Return whether or not the thread is currently running.*************************************
      bool isAlive (void) const
      {
         return m_thread_alive;
      }

#ifndef WINDOWS
      // Get the id of the thread.******************************************************************
      int getThreadId (void) const
      {
			//return (static_cast <int> (m_thread));
    	  return 0;
      }
#endif

#ifdef WINDOWS
	  // Get the id of the thread.******************************************************************
	  DWORD getThreadId (void) const
	  {
		  return GetThreadId (m_thread);
	  }
#endif

      // Set the thread's priority.*****************************************************************
      int setPriority (const int priority)
      {
         int return_value = ALREADY_RUNNING;
         if (!m_thread_alive)
         {
            return_value = COULD_NOT_SET_PRIORITY;
		    #ifdef WINDOWS
				if (SetThreadPriority (m_thread, priority))
				{
					return_value = SUCCESSFUL;
				}
		    #else
				struct sched_param param;
				// Get the current thread scheduling.
				pthread_attr_getschedparam (&m_thread_attributes, &param);
				param.sched_priority = static_cast <int> (priority);
				if (pthread_attr_setschedparam (&m_thread_attributes, &param) == 0)
				{
				   return_value = SUCCESSFUL;
				}
		    #endif
         }

         return return_value;
      }
      
#ifndef WINDOWS
      // Set the thread's scheduling policy.  Valid pthread inputs are SCHED_FIFO, SCHED_OTHER, or SCHED
      int setSchedulingPolicy (SchedPolicy policy)
      {
      	 int return_value = ALREADY_RUNNING;
		 policy = policy;
         if (!m_thread_alive)
         {
         	// Determine the pthread scheduling equivalent of the passed in value.
         	int policy = -1;
         	switch (policy)
         	{
         		case FIFO:
         		{
         			policy = SCHED_FIFO;
         		}
         		break;
         		
         		case ROUND_ROBIN:
         		{
         			policy = SCHED_RR;
         		}
         		break;
         		
         		case AUTO:
         		{
         			policy = SCHED_OTHER;
         		}
         		break;
         		
         		default:
         		{
         			return_value = UNKNOWN_SCHED_POLICY;
         		}
         		break;
         	}
         	
         	if (policy != -1)
         	{         	
            	return_value = COULD_NOT_SET_POLICY;
		         if (pthread_attr_setschedpolicy (&m_thread_attributes, policy) == 0)
		         {
		            return_value = SUCCESSFUL;
		         }
            }
         }

         return return_value;
      }
#endif

      // Get the thread's priority.****************************************************************
      int getPriority (void) const
      {
		 #ifdef WINDOWS
			 return GetThreadPriority (m_thread);
		 #else
			 struct sched_param param;

			 pthread_attr_getschedparam (&m_thread_attributes, &param);
			 return param.sched_priority;
		 #endif
      }

#ifndef WINDOWS
      // Set the stack size for the thread.*******************************************************
      int setStackSize (size_t size)
      {
         int return_value = ALREADY_RUNNING;
         if (!m_thread_alive)
         {
            return_value = SUCCESSFUL;
            pthread_attr_setstacksize (&m_thread_attributes, size);
         }

         return return_value;
      }
#endif

      // Operator overloads***********************************************************************

      // Operator=.*******************************************************************************
      const Thread & operator= (const Thread &rhs)
      {
         if (this != &rhs)
         {
            m_start_delegate = rhs.m_start_delegate;
            m_end_delegate = rhs.m_end_delegate;
            m_thread_alive = rhs.m_thread_alive;
            m_thread = rhs.m_thread;
			#ifndef WINDOWS
				m_thread_attributes = rhs.m_thread_attributes;
			#endif
         }
         return *this;
      }

      // Operator==.******************************************************************************
      bool operator== (const Thread &rhs) const
      {
		  #ifdef WINDOWS
			  return ((m_start_delegate == rhs.m_start_delegate)       &&
					  (m_end_delegate == rhs.m_end_delegate)           &&
					  (m_thread_alive == rhs.m_thread_alive)           &&
					  (m_thread == rhs.m_thread));
	     #else
			  return ((m_start_delegate == rhs.m_start_delegate)       &&
					  (m_end_delegate == rhs.m_end_delegate)           &&
					  (m_thread_alive == rhs.m_thread_alive)           &&
					  (pthread_equal (m_thread, rhs.m_thread) == 0));
	     #endif
      }

      // Operator!=.*****************************************************************************
      bool operator!= (const Thread &rhs) const
      {
         #ifdef WINDOWS
			  return ((m_start_delegate != rhs.m_start_delegate)       &&
					  (m_end_delegate != rhs.m_end_delegate)           &&
					  (m_thread_alive != rhs.m_thread_alive)           &&
					  (m_thread != rhs.m_thread));
	     #else
			  return ((m_start_delegate != rhs.m_start_delegate)       &&
					  (m_end_delegate != rhs.m_end_delegate)           &&
					  (m_thread_alive != rhs.m_thread_alive)           &&
					  (pthread_equal (m_thread, rhs.m_thread) != 0));
	     #endif    
      }

      /**
 	    * Sleep thread until a contional has signaled it to awaken.
	    */
      void conditionalWait (ThreadMutex &mutex)
      {
      	 #ifdef WINDOWS
		 #else
		    mutex.lock ();
		 	pthread_cond_wait (&m_conditional, mutex.getMutex ());
		 #endif
      }

	  /**
		* Signal thread a thread to awaken.
		*/
	  void signal (void)
	  {	
	  	#ifdef WINDOWS
		#else
			pthread_cond_signal (&m_conditional);	
		#endif
	  }

	  /**
		* Broadcast all threads to awaken.
		*/
	  void broadcast (void)
	  {
	  	#ifdef WINDOWS
		#else
			pthread_cond_broadcast (&m_conditional);
		#endif
	  }

   private:

      // Actually run the thread.  This will call the delegate that will call the correct member
      // function of the class used.  pthread requires a static or C-style function to call.
#ifndef WINDOWS
      static void *run (void *object)
      {
         // Get the correct instance to use.
         Thread *this_instance = reinterpret_cast <Thread *> (object);
         // Ensure that the object is valid.  It is unlikely that it won't be but consider
         // this to be a cautionary sanity check.
         if (this_instance)
         {
            this_instance->m_start_delegate (this_instance->m_param); // Use the delegate to call the correct function.
			if (this_instance->m_end_delegate)
			{
				this_instance->m_end_delegate (this_instance->m_end_param);
			}
         }

         this_instance->m_thread_alive = false;     		      	  // The thread has finished executing.
	 	 pthread_exit (NULL);
      }
#else
      static DWORD run (void *object)
      {
          // Get the correct instance to use.
          Thread *this_instance = reinterpret_cast <Thread *> (object);
          // Ensure that the object is valid.  It is unlikely that it won't be but consider
          // this to be a cautionary sanity check.
          if (this_instance)
          {
             this_instance->m_start_delegate (this_instance->m_param); // Use the delegate to call the correct function.
			 if (this_instance->m_end_delegate)
			 {
			 	this_instance->m_end_delegate (this_instance->m_end_param);
			 }
          }

          this_instance->m_thread_alive = false;     		      	  // The thread has finished executing.
		  return 0;
      }
#endif

      ThreadStart m_start_delegate;       // Stored delegate to use when the thread is started.
      ThreadEnd m_end_delegate;           // Stored delegate to use when the thread has finished.
      bool m_thread_alive;                // Determine if the thread is running or not.
      void *m_param;			  		  // Parameter to pass to the called function.
	  void *m_end_param;				  // Parameter to pass from the thread for the end of the function.

#ifdef WINDOWS
	  HANDLE m_thread;
#else
	  pthread_t m_thread;                 // The actual thread to run.
      pthread_attr_t m_thread_attributes; // Attributes of the thread.
	  pthread_cond_t m_conditional;
#endif
};



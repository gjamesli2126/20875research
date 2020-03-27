/*

   Filename : ThreadManager.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Manage a list of threads for the operating system (singleton).

   Change List:

      - 08/24/2010  - Created (Cody White)

*/

#pragma once

#include <thread/Thread.h>
#include <vector>

class ThreadManager
{
	public:

		/**
		  * Static function for getting an instance of this class.
		  */
		static ThreadManager *create (void);

		/**
		  * Remove a reference to this class.  Once the number of references has reached
		  * 0, the ThreadManager can delete itself.
		  */
		static void destroy (void);

		/**
		  * Get a thread for general use.
		  */
		Thread *getThread (void);

	private:

		/**
		  * Default constructor.
		  */
		ThreadManager (void);

		/**
		  * Destructor.
		  */
		~ThreadManager (void);

		/**
		  * Function the threads call when they're done executing.
		  * id Id of the thread that has finished.
		  */
		void threadDone (void *id);

		// Member variables.
		static ThreadManager *m_instance;			// Static instance of this class.
		int 				 m_num_threads;			// Number of threads that can be used.
		Thread				 *m_threads;			// List of threads that can be used.
		int					 m_references;			// Number of references to this class.
		ThreadMutex			 m_mutex;				// Mutex for locking.
		ThreadMutex			 m_signal_mutex;		// Signal mutex.
		std::vector <int>	 m_idle_threads;		// Threads that are idle.
		pthread_cond_t		 m_thread_done_signal;	// Signal of a thread finishing.
};


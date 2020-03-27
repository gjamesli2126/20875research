/*

   Filename : ThreadManager.cpp
   Author   : Cody White
   Version  : 1.0

   Purpose  : Manage a list of threads for the operating system (singleton).

   Change List:

      - 08/24/2010  - Created (Cody White)

*/

#include <thread/ThreadManager.h>
#include <iostream>
using namespace std;

ThreadManager *ThreadManager::m_instance = NULL;

// Default constructor.
ThreadManager::ThreadManager (void)
{
	// Get the number of cores available on this system.
	m_num_threads = sysconf (_SC_NPROCESSORS_ONLN);

	// Create the list of threads.
	m_threads = new Thread[m_num_threads];

	// Set the end parameters of all threads to be their index in the thread list.
	// Once a thread has finished executing, it will simply tell this class which one
	// it is so that the ThreadManager can keep track of all idle threads.
	for (int i = 0; i < m_num_threads; ++i)
	{
		m_threads[i].setEndParameter ((void *)i);
		m_threads[i].setEndFunction (ThreadEnd (this, &ThreadManager::threadDone));
		m_idle_threads.push_back (i);
	}

	m_instance = NULL;
	m_references = 0;
	pthread_cond_init (&m_thread_done_signal, NULL);
}

// Destructor.
ThreadManager::~ThreadManager (void)
{
	if (m_threads)
	{
		delete [] m_threads;
		m_threads = NULL;
	}

	m_num_threads = 0;
	m_instance = NULL;
}

// Static function for getting an instance of this class.
ThreadManager *ThreadManager::create (void)
{
	// Create a new instance of the class if one hasn't been yet.
	if (m_instance == NULL)
	{
		m_instance = new ThreadManager;
	}
	
	m_instance->m_references++;
	return m_instance;
}

// Remove a reference to this class.
void ThreadManager::destroy (void)
{
	m_instance->m_references--;
	if (m_instance->m_references == 0)
	{
		// There are no more refernces to this class, so delete it.
		delete m_instance;
	}
}

// Get a thread for general use.
Thread *ThreadManager::getThread (void)
{
	if (m_idle_threads.size () == 0)
	{
		// There are no threads available, so wait for one to become available.
		m_signal_mutex.lock ();
		pthread_cond_wait (&m_thread_done_signal, m_signal_mutex.getMutex ());
		m_signal_mutex.unlock ();
		// We've now been signaled that there is an available thread, move on.
	}
		
	// There are idle threads available for use, grab the front of the list.
	int id = m_idle_threads.front ();
	m_idle_threads.erase (m_idle_threads.begin ());
	return &(m_threads[id]);
}

// Function the threads call when they're done executing.
void ThreadManager::threadDone (void *id)
{
	// Add the thread to the idle threads list.
	m_mutex.lock ();
	m_idle_threads.push_back ((size_t)id);
	// Signal that there is a thread in the queue in case the getThread ()
	// function is waiting for one.
	pthread_cond_signal (&m_thread_done_signal);
	m_mutex.unlock ();
}


/*
   Filename : ContextBuffer.h
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Purpose  : Encapsulates context-specific data. 

   Change List:

      - 06/18/2009  - Created (Cody White and Joe Mahsman)
*/

#pragma once

#include <map>
#include <thread/Thread.h>
#include <iostream>

namespace gfx
{

/**
 * Thread-safe buffer for data associated with multiple rendering contexts.
 *
 * Contexts are identified by an integer ID. An integer is used for its
 * terse simplicity. Although unsigned int would give more than 65,535
 * possible contexts, this many contexts are never needed (at least not
 * until ubiquitous VR (tm) circa 2200 AD).
 *
 * The template parameter T defines the type of data stored per context ID.
 * This can be anything from a built-in type (i.e. int, float) to a struct
 * or class.
 */
template <class T>
class ContextBuffer
{
	public:

		/**
		  * Default constructor.
		  */
		ContextBuffer (void);

		/**
		  * Destructor.
		  */
		~ContextBuffer (void);

		/**
		 * Return a reference to context data corresponding to the context ID.
		 * If the ID does not exist, a new context is created and a reference
		 * to it is returned.
		 * @param id Context-id to create/use.
		 */
		inline T& operator[] (int id);

		/**
		 * const version of operator[].
		 * The same functionality is provided, except that a new context is
		 * NOT created when the search fails.
		 * @param id Context-id to use.
		 */
		inline const T& operator[] (int id) const;

		/**
		  * Get the size of the context buffer.
		  */
		inline size_t size (void) const;

		/**
		  * Remove an element from the context buffer.
		  * @param id Context-id element to remove.
		  */
		inline void remove (int id);

		/**
		  * Clear the context buffer.
		  */
		inline void clear (void);

		// ContextBuffer iterator.
		typedef typename std::map <int, T>::iterator 	   ContextBufferIterator;
		typedef typename std::map <int, T>::const_iterator ConstContextBufferIterator;

		/**
		  * Find an element in the buffer.
		  * @param element Element to find.
		  */
		inline ContextBufferIterator find (int element)  
		{
			return m_data.find (element);
		}

		/**
		  * Find an element in the buffer. const version.
		  * @param element Element to find.
		  */
		inline ConstContextBufferIterator find (int element) const
		{
			return m_data.find (element);
		}

		/**
		  * Get the beginning iterator to the data in the context buffer.
		  */
		ContextBufferIterator begin (void)
		{
			return m_data.begin ();
		}

		/**
		  * Get the end iterator to the data in the context buffer.
		  */
		ContextBufferIterator end (void)
		{
			return m_data.end ();
		}

		ConstContextBufferIterator begin (void) const
		{
			return m_data.begin ();
		}

		ConstContextBufferIterator end (void) const
		{
			return m_data.end ();
		}

	private:

		// Member variables.
		mutable ThreadMutex m_mutex;
		std::map<int, T> m_data;
};

template <class T>
ContextBuffer<T>::ContextBuffer (void)
{
}

template <class T>
ContextBuffer<T>::~ContextBuffer (void)
{
}

template <class T>
inline T& ContextBuffer<T>::operator[] (int id)
{
	m_mutex.lock ();
	T& result = m_data[id];
	m_mutex.unlock ();

	return result;
}

template <class T>
inline const T& ContextBuffer<T>::operator[] (int id) const 
{
	m_mutex.lock ();
	const T& result = m_data.find(id)->second;
	m_mutex.unlock ();

	return result;
}

template <class T>
inline void ContextBuffer<T>::remove (int id)
{
	m_mutex.lock ();
	ContextBufferIterator iter = m_data.find (id);
	m_data.erase (iter);
	m_mutex.unlock ();
}


template <class T>
inline size_t ContextBuffer<T>::size (void) const
{
	size_t tmp = 0;
	m_mutex.lock ();
	tmp = m_data.size ();
	m_mutex.unlock ();

	return tmp;
}

template <class T>
inline void ContextBuffer<T>::clear (void)
{
	m_mutex.lock ();
	m_data.clear ();
	m_mutex.unlock ();
}

}

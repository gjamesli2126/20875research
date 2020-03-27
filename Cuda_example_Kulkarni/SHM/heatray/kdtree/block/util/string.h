/*

   Filename : string.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : This class implements both the standard C++ string and
   			  stringstream class as well as some newly added functionality
			  into one easy-to-use class.

   Change List:

      - 05/15/2009  - Created (Cody White)
*/

#pragma once

#ifndef __STRING_H__
#define __STRING_H__

// Include string and stringstream headers
#include <string>
#include <sstream>
#include <iostream>

namespace util
{

// Main class definition
class string : public std::string // (super string)
{
	public:

		/**
		  * Operator<<
		  * Automatically appends whatever is passed to it as a string.
		  */
		template <typename T>
		string & operator<< (const T &v)
		{
			std::stringstream ss;
			ss << v;
			(*this) += ss.str ();
			return *this;
		}

		/**
		  * Convert the string into whatever type it needs to be.
		  */
		template <typename T>
		const string & operator>> (T &v) const
		{
			std::stringstream ss (*this);
			ss >> v;
			return *this;
		}

		/**
		  * Convert whatever is passed in and store it as a string.
		  */
		template <typename T>
		string (const T &v)
		{
			*this << v;
		}

		/**
		  * Cast the string into whatever.
		  */
		template <typename T>
		operator T (void)
		{
			T v;
			*this >> v;
			return v;
		}

		/**
		  * Empty constructor.
		  */
		string (void)
		{
			*this =  "";
		}

		/**
		  * Determine if a string ends with a given substring.
		  */
		bool endsWith (string end)
		{
			size_t index = (*this).find (end);
			size_t size_of_end = end.size ();
			size_t size = (*this).size ();

			return (index + (size_of_end) == size);
		}
};

}
#endif



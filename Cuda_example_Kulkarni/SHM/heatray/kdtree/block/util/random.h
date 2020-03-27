#pragma once

#include <cstdlib>

#include <math/Vector.h>
#include "boost/random.hpp"

/// Uniform distribution random numbers
typedef boost::rand48 base_generator_type;

class RandomFloat {
	base_generator_type generator;
	boost::uniform_real<float> uni_dist;
	boost::variate_generator<base_generator_type, boost::uniform_real<float> > gen;
public:
	RandomFloat(uint64_t seed) : generator(seed), uni_dist(0.0f,1.0f), gen(generator,uni_dist) { }
	//Ran(uint64_t seed, int min, int max) : generator((uint64_t)seed), uni_dist(min,max), gen(generator,uni_dist) { }

	float get() { return gen(); } ///< Generate a random number

	float interval(float fMin, float fMax) {
		float fUnit = gen();
		float fDiff = fMax - fMin;
		return fMin + fUnit * fDiff;
	}
};

//namespace util
//{
//
//inline int random(int nMin, int nMax)
//{
//	int nDiff = nMax - nMin + 1;
//
//	return nMin + (rand( ) % nDiff);
//}
//
//
//inline float random(float fMin, float fMax)
//{
//	float fUnit = float(rand( )) / RAND_MAX;
//	float fDiff = fMax - fMin;
//
//	return fMin + fUnit * fDiff;
//}
//
//inline int randomSign()
//{
//	if (random(0, 100) < 50)
//	{
//		return 1;
//	}
//
//	return -1;
//}
//
//template <class T, unsigned int N>
//inline math::Vector<T,N> random(const math::Vector<T,N>& a, const math::Vector<T,N>& b)
//{
//	math::Vector<T,N> result;
//
//	for (unsigned int i = 0; i < N; ++i)
//	{
//		result[i] = random(a[i], b[i]);
//	}
//
//	return result;
//}
//
//}

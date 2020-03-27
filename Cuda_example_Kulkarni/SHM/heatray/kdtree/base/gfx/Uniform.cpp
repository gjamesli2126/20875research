#include "gfx/Uniform.h"

namespace gfx
{

Uniform::Uniform()
{
	_pid = 0;
	_location = -1;
}

Uniform::Uniform(GLuint pid, const char* name)
{
#ifdef USE_GL
	_pid = pid;
	_location = glGetUniformLocation(_pid, name);

	if (_location == -1)
	{
		throw("Could not find uniform.");
	}
#endif
}

void Uniform::set1i(int i) const
{
#ifdef USE_GL
	glUniform1i(_location, i);
#endif
}

void Uniform::set1f(float f) const
{
#ifdef USE_GL
	glUniform1f(_location, f);
#endif
}

int Uniform::get1i() const
{
#ifdef USE_GL
	int i;
	glGetUniformiv(_pid, _location, &i);
	return i;
#else
	return 0;
#endif
}

float Uniform::get1f() const
{
#ifdef USE_GL
	int f;
	glGetUniformiv(_pid, _location, &f);
	return f;
#else
	return 0.f;
#endif
}

}

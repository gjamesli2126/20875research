#pragma once

#include <GL/glew.h>

namespace gfx
{

/**
 * Encapsulates a GLSL uniform variable.
 * A uniform location remains valid until the program is relinked.
 */
class Uniform
{
public:
	Uniform();
	Uniform(GLuint pid, const char* name);

	void set1i(int i) const;
	void set1f(float f) const;

	int get1i() const;
	float get1f() const;
	
private:
	GLint _location;
	GLuint _pid;
};

}

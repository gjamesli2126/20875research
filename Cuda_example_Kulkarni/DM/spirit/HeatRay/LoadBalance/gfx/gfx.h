// Facade header to ensure that glew is always included first.
// In addition you get some debugging functions.

#include <GL/glew.h>

#ifdef USE_GL
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif
#endif

#include <gfx/util.h>

#include <iostream>

using namespace std;

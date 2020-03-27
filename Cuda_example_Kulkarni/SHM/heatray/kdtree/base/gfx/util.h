#pragma once

#include "gfx.h"
#include <stdexcept>
#include <math/Vector.h>

namespace gfx
{

inline void translatefv(math::vec3f& pos)
{
	glTranslatef(pos[0], pos[1], pos[2]);
}

inline void scalef(float scale)
{
	glScalef(scale, scale, scale);
}

inline void checkGLErrors(const char* tag = "")
{
	GLenum errorID = glGetError();

	if (errorID != GL_NO_ERROR)
	{
		//const char *errorString = (const char*)gluErrorString(errorID);
		const char *errorString = (const char*)"";

		printf("GL error: %s\n", errorString);

		if (tag != NULL)
		{
			printf("tag: %s\n", tag);
		}

		throw std::runtime_error(errorString);
	}
}

inline void renderAxes()
{
	glBegin(GL_LINES);
		glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(1, 0, 0);
		glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, 1, 0);
		glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, 1);
	glEnd();
}

inline void renderCube (float size = 1.0)
{
	math::vec3f dimensions = math::vec3f(size, size, size);
	math::vec3f origin = math::vec3f(0.0f, 0.0f, 0.0f) - dimensions * 0.5f;
	math::vec3f corner = origin + dimensions;

	math::vec3f point0 = math::vec3f(origin[0], origin[1], origin[2], 1.0);
	math::vec3f point1 = math::vec3f(corner[0], origin[1], origin[2], 1.0f);
	math::vec3f point2 = math::vec3f(origin[0], corner[1], origin[2], 1.0f);
	math::vec3f point3 = math::vec3f(corner[0], corner[1], origin[2], 1.0f);
	math::vec3f point4 = math::vec3f(origin[0], origin[1], corner[2], 1.0f);
	math::vec3f point5 = math::vec3f(corner[0], origin[1], corner[2], 1.0f);
	math::vec3f point6 = math::vec3f(origin[0], corner[1], corner[2], 1.0f);
	math::vec3f point7 = math::vec3f(corner[0], corner[1], corner[2], 1.0f);

	glMatrixMode(GL_MODELVIEW);
	glBegin(GL_QUADS);
	
	// Back
	glTexCoord2f (0.0f, 1.0f);	glNormal3f (0.0f, 0.0f, -1.0f);	glVertex3fv(point4.v);
	glTexCoord2f (1.0f, 1.0f);	glNormal3f (0.0f, 0.0f, -1.0f);	glVertex3fv(point6.v);
	glTexCoord2f (1.0f, 0.0f);	glNormal3f (0.0f, 0.0f, -1.0f);	glVertex3fv(point2.v);
	glTexCoord2f (0.0f, 0.0f);	glNormal3f (0.0f, 0.0f, -1.0f);	glVertex3fv(point0.v);

	// Right
	glTexCoord2f (0.0f, 1.0f);	glNormal3f (1.0f, 0.0f, 0.0f);	glVertex3fv(point1.v);
	glTexCoord2f (1.0f, 1.0f);	glNormal3f (1.0f, 0.0f, 0.0f);	glVertex3fv(point3.v);
	glTexCoord2f (1.0f, 0.0f);	glNormal3f (1.0f, 0.0f, 0.0f);	glVertex3fv(point7.v);
	glTexCoord2f (0.0f, 0.0f);	glNormal3f (1.0f, 0.0f, 0.0f);	glVertex3fv(point5.v);

	// Bottom
	glTexCoord2f (0.0f, 1.0f);	glNormal3f (0.0f, -1.0f, 0.0f);	glVertex3fv(point4.v);
	glTexCoord2f (1.0f, 1.0f);	glNormal3f (0.0f, -1.0f, 0.0f);	glVertex3fv(point0.v);
	glTexCoord2f (1.0f, 0.0f);	glNormal3f (0.0f, -1.0f, 0.0f);	glVertex3fv(point1.v);
	glTexCoord2f (0.0f, 0.0f);	glNormal3f (0.0f, -1.0f, 0.0f);	glVertex3fv(point5.v);

	// Top
	glTexCoord2f (0.0f, 1.0f);	glNormal3f (0.0f, 1.0f, 0.0f);	glVertex3fv(point6.v);
	glTexCoord2f (1.0f, 1.0f);	glNormal3f (0.0f, 1.0f, 0.0f);	glVertex3fv(point7.v);
	glTexCoord2f (1.0f, 0.0f);	glNormal3f (0.0f, 1.0f, 0.0f);	glVertex3fv(point3.v);
	glTexCoord2f (0.0f, 0.0f);	glNormal3f (0.0f, 1.0f, 0.0f);	glVertex3fv(point2.v);

	// Left
	glTexCoord2f (0.0f, 1.0f);	glNormal3f (-1.0f, 0.0f, 0.0f);	glVertex3fv(point0.v);
	glTexCoord2f (1.0f, 1.0f);	glNormal3f (-1.0f, 0.0f, 0.0f);	glVertex3fv(point2.v);
	glTexCoord2f (1.0f, 0.0f);	glNormal3f (-1.0f, 0.0f, 0.0f);	glVertex3fv(point3.v);
	glTexCoord2f (0.0f, 0.0f);	glNormal3f (-1.0f, 0.0f, 0.0f);	glVertex3fv(point1.v);

	// Front
	glTexCoord2f (0.0f, 1.0f);	glNormal3f (0.0f, 0.0f, 1.0f);	glVertex3fv(point5.v);
	glTexCoord2f (1.0f, 1.0f);	glNormal3f (0.0f, 0.0f, 1.0f);	glVertex3fv(point7.v);
	glTexCoord2f (1.0f, 0.0f);	glNormal3f (0.0f, 0.0f, 1.0f);	glVertex3fv(point6.v);
	glTexCoord2f (0.0f, 0.0f);	glNormal3f (0.0f, 0.0f, 1.0f);	glVertex3fv(point4.v);

	glEnd();
}

}

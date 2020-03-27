/*
   Filename : Skybox.h
   Author   : Joe Mahsman
   Version  : 1.0

   Purpose  : Implements a easy-to-use interface to create a skybox around the camera.

   Change List:

      - 06/12/2009  - Created (Joe Mahsman)
*/

#include <util/file.h>
#include <iostream>

#include <gfx/Skybox.h>
#include <util/file.h>

using namespace std;
using namespace math;
using util::fileExists;

namespace gfx
{

// Default constructor
Skybox::Skybox (void)
{
	//_position		 = math::vec3f (0.0f, 0.0f, 0.0f);
	_scale           = 1.0f;
}

// Destructor
Skybox::~Skybox (void)
{
}

// Initilize the skybox with a directory to load image files from.
void Skybox::load (const util::string& dir)
{
	util::string type;

	// Determine the file type to use.
	if (filesExist(dir, "jpg"))
	{
		type = "jpg";
	}
	else if (filesExist(dir, "png"))
	{
		type = "png";
	}
	else if (filesExist(dir, "tga"))
	{
		type = "tga";
	}
	else if (filesExist(dir, "bmp"))
	{
		type = "bmp";
	}

	if (type.empty())
	{
		cout << "Skybox::load () - Error: Could not find any suitable files in " << dir << "\n";
		exit(1);
	}

	util::string prefix = dir + "/";
	util::string suffix = "." + type;

	gfx::Texture::Params p;
	p.wrapS = GL_CLAMP; // removes seams
	p.wrapT = GL_CLAMP;
	p.envMode = GL_DECAL;
	//p.minFilter = GL_LINEAR; // experiment with these
	//p.magFilter = GL_LINEAR;
	for (int i = 0; i < 6; ++i) _images[i].setParams(p);

	// Load the images.
	_images[0].load((prefix + "posx" + suffix).c_str());
	_images[1].load((prefix + "negx" + suffix).c_str());
	_images[2].load((prefix + "posy" + suffix).c_str());
	_images[3].load((prefix + "negy" + suffix).c_str());
	_images[4].load((prefix + "posz" + suffix).c_str());
	_images[5].load((prefix + "negz" + suffix).c_str());
}

// Create the new graphics context.
void Skybox::initContext(int context_id)
{
	_images[0].initContext(context_id);
	_images[1].initContext(context_id);
	_images[2].initContext(context_id);
	_images[3].initContext(context_id);
	_images[4].initContext(context_id);
	_images[5].initContext(context_id);
}

// Remove the created graphics context.
void Skybox::destroyContext (int context_id)
{
	_images[0].destroyContext(context_id);
	_images[1].destroyContext(context_id);
	_images[2].destroyContext(context_id);
	_images[3].destroyContext(context_id);
	_images[4].destroyContext(context_id);
	_images[5].destroyContext(context_id);
}

// Render the skybox around the camera.
void Skybox::render (int context_id)
{
#ifdef USE_GL
	vec3f& pos = _positions[context_id];

	// Based on http://sidvind.com/wiki/Skybox_tutorial

	glPushMatrix();

	// Position the box at the camera's current position
	glTranslatef(pos[0], pos[1], pos[2]);
	glScalef(_scale, _scale, _scale);

	glPushAttrib(GL_ENABLE_BIT);

	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);

	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK); // front faces

	// Useful if textures are not working
	//glColor4f(1, 0, 1, 1);
	//glColor4f(1, 1, 1, 1); //TODO: fix this

	// The skybox is rendered with respect to the default OpenGL
	// left-handed coordinate system and camera (down the -z axis)

	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

	// Render the right quad (posx)
	_images[0].bind(context_id);
	glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex3f(  0.5f, -0.5f, -0.5f );
		glTexCoord2f(1, 0); glVertex3f(  0.5f, -0.5f,  0.5f );
		glTexCoord2f(1, 1); glVertex3f(  0.5f,  0.5f,  0.5f );
		glTexCoord2f(0, 1); glVertex3f(  0.5f,  0.5f, -0.5f );
	glEnd();

	// Render the left quad (negx)
	_images[1].bind(context_id);
	glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex3f( -0.5f, -0.5f,  0.5f );
		glTexCoord2f(1, 0); glVertex3f( -0.5f, -0.5f, -0.5f );
		glTexCoord2f(1, 1); glVertex3f( -0.5f,  0.5f, -0.5f );
		glTexCoord2f(0, 1); glVertex3f( -0.5f,  0.5f,  0.5f );
	glEnd();
	
	// Render the top quad (posy)
	_images[2].bind(context_id);
	glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex3f(  0.5f,  0.5f, -0.5f );
		glTexCoord2f(1, 0); glVertex3f(  0.5f,  0.5f,  0.5f );
		glTexCoord2f(1, 1); glVertex3f( -0.5f,  0.5f,  0.5f );
		glTexCoord2f(0, 1); glVertex3f( -0.5f,  0.5f, -0.5f );
	glEnd();

	// Render the bottom quad (negy)
	_images[3].bind(context_id);
	glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex3f( -0.5f, -0.5f, -0.5f );
		glTexCoord2f(1, 0); glVertex3f( -0.5f, -0.5f,  0.5f );
		glTexCoord2f(1, 1); glVertex3f(  0.5f, -0.5f,  0.5f );
		glTexCoord2f(0, 1); glVertex3f(  0.5f, -0.5f, -0.5f );
	glEnd();

	// Render the back quad (posz)
	_images[4].bind(context_id);
	glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex3f(  0.5f, -0.5f,  0.5f );
		glTexCoord2f(1, 0); glVertex3f( -0.5f, -0.5f,  0.5f );
		glTexCoord2f(1, 1); glVertex3f( -0.5f,  0.5f,  0.5f );
		glTexCoord2f(0, 1); glVertex3f(  0.5f,  0.5f,  0.5f );
	glEnd();

	// Render the front quad (negz)
	_images[5].bind(context_id);
	glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex3f( -0.5f, -0.5f, -0.5f );
		glTexCoord2f(1, 0); glVertex3f(  0.5f, -0.5f, -0.5f );
		glTexCoord2f(1, 1); glVertex3f(  0.5f,  0.5f, -0.5f );
		glTexCoord2f(0, 1); glVertex3f( -0.5f,  0.5f, -0.5f );
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	glPopAttrib();
	glPopMatrix();	
#endif
}

// Determine if files of a given type exist in the passed-in directory.
bool Skybox::filesExist (const util::string& dir, const util::string& type)
{
	return fileExists(dir + "/posx." + type) &&
	       fileExists(dir + "/negx." + type) &&
	       fileExists(dir + "/posy." + type) &&
	       fileExists(dir + "/negy." + type) &&
	       fileExists(dir + "/posz." + type) &&
	       fileExists(dir + "/negz." + type) ;
}

void Skybox::setPosition (const math::vec3f& position, int id)
{
	_positions[id] = position;
}

void Skybox::setPosition (const float* position, int id)
{
	vec3f& pos = _positions[id];

	pos[0] = position[0];
	pos[1] = position[1];
	pos[2] = position[2];
}

}

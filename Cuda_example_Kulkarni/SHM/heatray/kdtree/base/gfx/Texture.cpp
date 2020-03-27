/*
   Filename : Texture.cpp
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Purpose  : Encapsulate OpenGL texture calls. 

   Change List:

      - 06/18/2009  - Created (Cody White and Joe Mahsman)

	  - 09/16/2009  - Changed to use FreeImage instead of DevIL.
*/

#include <gfx/Texture.h>

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <gfx/util.h>
#include <cmath>

using namespace std;

#define VERBOSE

namespace gfx
{

Texture::Texture (void)
{
	m_width = 0;
	m_height = 0;
	m_data = NULL;
	texture_name = "";
}

Texture::~Texture (void)
{
#ifdef USE_GL
	gfx::ContextBuffer<GLuint>::ContextBufferIterator iter;
	for (iter = m_ids.begin ();
		 iter != m_ids.end ();
		 ++iter)
	{
		glDeleteTextures (1, &(iter->second));
	}

	m_ids.clear ();
#endif
}

void Texture::destroyContext (int context_id)
{
#ifdef USE_GL
	glDeleteTextures (1, &m_ids[context_id]);
	m_ids.remove (context_id);
#endif
}

void Texture::load (const char* filename)
{
	FREE_IMAGE_FORMAT format = FreeImage_GetFileType (filename, 0);
	FIBITMAP* unconvertedData = FreeImage_Load (format, filename);
	if (unconvertedData == NULL)
	{
		stringstream error;
		error << "Texture::load(): Unable to load \"" << filename << "\".\n";
		throw runtime_error (error.str ());
	}

	m_data = FreeImage_ConvertTo32Bits (unconvertedData);
	FreeImage_Unload (unconvertedData);
	m_width = FreeImage_GetWidth (m_data);
	m_height = FreeImage_GetHeight (m_data);
}

void Texture::initContext (int context_id)
{
#ifdef USE_GL
	GLuint& id = m_ids[context_id];

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_2D, id);

	applyParams ();  

	// Get the maximum texture size for this GPU.
	GLint max_size = 0;
	glGetIntegerv (GL_MAX_TEXTURE_SIZE, &max_size);
	if (m_width > max_size || m_height > max_size)
	{
		gluBuild2DMipmaps(GL_TEXTURE_2D,
	                  m_params.internalFormat,
	                  m_width, m_height,
	                  GL_BGRA,
	                  GL_UNSIGNED_BYTE,
	                  FreeImage_GetBits (m_data));
		checkGLErrors("Texture::createTextureObject1 - gluBuild2DMipmaps");
	}
	else
	{
		glTexImage2D (GL_TEXTURE_2D, 0, m_params.internalFormat, m_width, m_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, FreeImage_GetBits (m_data));
		glGenerateMipmapEXT (GL_TEXTURE_2D);
		checkGLErrors("Texture::createTextureObject1 - glTexImage2D");
	}

	unbind ();
#endif
}

// Create a texture object with passed in data.
void Texture::initContext (void *data, math::vec2i size, GLenum data_type, int context_id)
{
#ifdef USE_GL
	m_width  = size.x ();
	m_height = size.y ();

	GLuint &id = m_ids[context_id];

	glEnable (GL_TEXTURE_2D);
	glGenTextures (1, &id);
	glBindTexture (GL_TEXTURE_2D, id);

	applyParams ();

	// Get the maximum texture size for this GPU.
	GLint max_size = 0;
	glGetIntegerv (GL_MAX_TEXTURE_SIZE, &max_size);
	if (m_width > max_size || m_height > max_size)
	{
		gluBuild2DMipmaps (GL_TEXTURE_2D,
				           m_params.internalFormat,
						   m_width, 
						   m_height,
						   m_params.format,
						   data_type,
						   data);
		checkGLErrors ("Texture::createTextureObject2 - gluBuild2DMipmaps");
	}
	else
	{
		glTexImage2D (GL_TEXTURE_2D, 
				      0, 
					  m_params.internalFormat, 
					  m_width, 
					  m_height, 
					  0, 
					  m_params.format, 
					  data_type, 
					  data);
		glGenerateMipmapEXT (GL_TEXTURE_2D);
		checkGLErrors("Texture::createTextureObject2 - glTexImage2D");
	}

	unbind ();
#endif
}

// Determine if this texture has loaded any data or not.
bool Texture::hasData (void) const
{
	return (m_data != NULL);
}

void Texture::clearCPUData (void)
{
	if (m_data)
	{
		FreeImage_Unload (m_data);
		m_data = NULL; 
	}
}

void Texture::blank (const size_t width, const size_t height, int data_type, int context_id)
{
#ifdef USE_GL
	// Get the maximum texture size for this GPU.
	GLint max_size = 0;
	glGetIntegerv (GL_MAX_TEXTURE_SIZE, &max_size);

	if (width > (GLuint)max_size || height  > (GLuint)max_size)
	{
		std::cout << "Texuture::blank () -- Error: image to large of this graphics card.  Ensure image dimensions are less than " << max_size << "x" << max_size << std::endl;
		return;
	}

	GLuint &id = m_ids[context_id];
	glGenTextures (1, &id);
	glBindTexture (GL_TEXTURE_2D, id);

	applyParams ();
	glTexImage2D (GL_TEXTURE_2D, 0, m_params.internalFormat, width, height, 0, m_params.format, data_type, NULL);
	glBindTexture (GL_TEXTURE_2D, 0);

	m_width = width;
	m_height = height;
#endif
}

void Texture::createMipmaps (int context_id)
{
#ifdef USE_GL
	if (valid (context_id))
	{
		bind (context_id);
		glGenerateMipmapEXT (GL_TEXTURE_2D);
		unbind ();
	}
#endif
}

void Texture::bind (int active_texture, int context_id) const 
{
#ifdef USE_GL
	GLuint id = m_ids[context_id];
	glEnable(GL_TEXTURE_2D);
	glActiveTexture (GL_TEXTURE0 + active_texture);
	glBindTexture(GL_TEXTURE_2D, id);
#endif
}

void Texture::unbind (int active_texture) const
{
#ifdef USE_GL
	glActiveTexture (GL_TEXTURE0 + active_texture);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
#endif
}

void Texture::applyParams()
{
#ifdef USE_GL
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, m_params.envMode);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, m_params.minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, m_params.magFilter);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, m_params.wrapS);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, m_params.wrapT);
#endif
}

math::Vector <float, 4> Texture::getPixelColor (float u, float v)
{
	math::Vector <float, 4> result = math::Vector <float, 4>::zero ();
	unsigned int x = std::ceil (u * m_width);
	unsigned int y = std::ceil (v * m_height);

	RGBQUAD color;
	if (FreeImage_GetPixelColor (m_data, x, y, &color))
	{
		float divide = 1.0f / 255.0f;
		result[0] = (float)color.rgbRed * divide;
		result[1] = (float)color.rgbGreen * divide;
		result[2] = (float)color.rgbBlue * divide;
		result[3] = (float)color.rgbReserved * divide;
	}

	return result;
}

GLuint & Texture::attachment (int context_id)
{
	return m_attachments[context_id];
}

const GLuint Texture::attachment (int context_id) const
{
	return m_attachments[context_id];
}

}


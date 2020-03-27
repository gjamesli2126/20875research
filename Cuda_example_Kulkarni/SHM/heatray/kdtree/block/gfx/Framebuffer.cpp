/*
   Filename : Framebuffer.cpp
   Author   : Cody White 
   Version  : 1.0

   Purpose  : Encapsulate OpenGL FBO calls. 

   Change List:

      - 01/27/2011  - Created (Cody White)
*/

#include <gfx/Framebuffer.h>
#include <gfx/util.h>
#include <iostream>

namespace gfx
{

// Default constructor.
Framebuffer::Framebuffer (void)
{
}

// Destructor.
Framebuffer::~Framebuffer (void)
{
#ifdef USE_GL
	gfx::ContextBuffer<GLuint>::ContextBufferIterator iter;
	for (iter = m_ids.begin ();
		 iter != m_ids.end ();
		 ++iter)
	{
		glDeleteFramebuffers (1, &(iter->second));
	}

	m_ids.clear ();
#endif
}

// Initialize the context for use.
void Framebuffer::initContext (int context_id)
{
#ifdef USE_GL
	// Get an id for this context.
	GLuint &id = m_ids[context_id];

	// Generate the FBO.
	glGenFramebuffers (1, &id);
#endif
}

// Destroy a context.
void Framebuffer::destroyContext (int context_id)
{
#ifdef USE_GL
	// Get the id for this context.
	GLuint &id = m_ids[context_id];

	// Destroy the framebuffer.
	glDeleteFramebuffers (1, &id);

	// Remove the context id from the context buffer.
	m_ids.remove (context_id);
#endif
}

// Attach a color texture to the FBO.  The FBO must be bound.
void Framebuffer::attachColor (unsigned int number, gfx::Texture &texture, int context_id)
{
#ifdef USE_GL
	glFramebufferTexture2D (GL_DRAW_FRAMEBUFFER, 
							GL_COLOR_ATTACHMENT0 + number,
							GL_TEXTURE_2D,
							texture.id (context_id),
							0); // Select the base mipmap level.
	texture.attachment (context_id) = number;
#endif
}

// Unattach a texture as a color.  
void Framebuffer::unattachColor (unsigned int number)
{
#ifdef USE_GL
	glFramebufferTexture2D (GL_DRAW_FRAMEBUFFER,
							GL_COLOR_ATTACHMENT0 + number,
							GL_TEXTURE_2D,
							0,
							0); // Mipmap level doesn't matter.
#endif
}

// Attach a depth buffer.
void Framebuffer::attachDepth (gfx::Texture &texture, int context_id)
{
#ifdef USE_GL
	glFramebufferTexture2D (GL_DRAW_FRAMEBUFFER,
							GL_DEPTH_ATTACHMENT,
							GL_TEXTURE_2D,
							texture.id (context_id),
							0);
#endif
}

// Bind the FBO for use.
void Framebuffer::bind (int context_id) const
{
#ifdef USE_GL
	glBindFramebuffer (GL_DRAW_FRAMEBUFFER, m_ids [context_id]);
#endif
}

// Unbind the FBO for use.
void Framebuffer::unbind (void) const
{
#ifdef USE_GL
	glBindFramebuffer (GL_DRAW_FRAMEBUFFER, 0);
#endif
}

// Set which buffer to draw to.
void Framebuffer::setDrawBuffer (unsigned int number) const
{
#ifdef USE_GL
	glDrawBuffer (GL_COLOR_ATTACHMENT0 + number);
#endif
}

// Set multiple draw buffers to use.
void Framebuffer::setMultipleDrawBuffers (unsigned int range) const
{
#ifdef USE_GL
	static const GLenum buffers[] = {
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2,
		GL_COLOR_ATTACHMENT3,
		GL_COLOR_ATTACHMENT4,
		GL_COLOR_ATTACHMENT5,
		GL_COLOR_ATTACHMENT6,
		GL_COLOR_ATTACHMENT7,
		GL_COLOR_ATTACHMENT8,
		GL_COLOR_ATTACHMENT9,
		GL_COLOR_ATTACHMENT10,
		GL_COLOR_ATTACHMENT11,
		GL_COLOR_ATTACHMENT12,
		GL_COLOR_ATTACHMENT13,
		GL_COLOR_ATTACHMENT14,
		GL_COLOR_ATTACHMENT15};

	glDrawBuffers (range, buffers);
#endif
}

// Show any errors in the FBO.
bool Framebuffer::checkError (void) const
{
#ifdef USE_GL
	switch (glCheckFramebufferStatusEXT (GL_FRAMEBUFFER_EXT))
	{
		case GL_FRAMEBUFFER_COMPLETE_EXT:
		{
			return true;
		}
		break;

		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
		{
			std::cout << "Framebuffer Error: Missing required image/buffer attachment" << std::endl;
		}
		break;

		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
		{
			std::cout << "Framebuffer Error: No buffers attached" << std::endl;
		}
		break;

		case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
		{
			std::cout << "Framebuffer Error: Mismatched dimensions" << std::endl;
		}
		break;

		case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
		{
			std::cout << "Framebuffer Error: Colorbuffers have different formats" << std::endl;
		}
		break;

		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
		{
			std::cout << "Framebuffer Error: Trying to draw to non-attached color buffer" << std::endl;
		}
		break;

		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
		{
			std::cout << "Framebuffer Error: Trying to read from non-attached read buffer" << std::endl;
		}
		break;

		case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
		{
			std::cout << "Framebuffer Error: Framebuffers not supported by graphics hardware" << std::endl;
		}
		break;
	}
#endif
	return false;
}

}


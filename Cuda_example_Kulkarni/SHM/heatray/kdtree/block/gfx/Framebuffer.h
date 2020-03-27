/*
   Filename : Framebuffer.h
   Author   : Cody White 
   Version  : 1.0

   Purpose  : Encapsulate OpenGL FBO calls. 

   Change List:

      - 01/27/2011  - Created (Cody White)
*/

#pragma once

#include <GL/glew.h>

#include <gfx/Texture.h>
#include <gfx/ContextBuffer.h>
#include <gfx/gfx.h>
#include <iostream>

namespace gfx
{

/**
 * Contains the definition for a framebuffer object in OpenGL.
 */
class Framebuffer
{
	public:

		/**
		  * Default constructor.
		  */
		Framebuffer (void);

		/**
		  * Destructor.
		  */
		~Framebuffer (void);

		/**
		 * Initialize the framebuffer for the current context.
		  * @param context_id ID for the current context.
		 */
		void initContext (int context_id);

		/**
		  * Destroy the context fbo.
		  * @param context_id ID to destroy.
		  */
		void destroyContext (int context_id);

		/**
		  * Add a texture as a color attachment. The framebuffer must be bound first.
		  * @param number Attachment number, between 0 and 15
		  * @param texture Texture to attach.
		  * @param context_id ID for this context.
		  */
		void attachColor (unsigned int number, gfx::Texture &texture, int context_id);

		/**
		  * Unattach a texture as a color.  The framebuffer must be bound.
		  * @param number the attachment number to unattach.
		  */
		void unattachColor (unsigned int number);

		/**
		  * Attach a depth buffer.
		  * @param depth Depth buffer to attach.
		  * @param context_id ID for this context.
		  */
		void attachDepth (gfx::Texture &texure, int context_id);

		/**
		  * Bind the frame buffer.
		  */
		void bind (int context_id) const;

		/**
		  * Unbind the framebuffer.
		  */
		void unbind (void) const;

		/**
		  * Set which buffer to draw to.
		  * @param buffer number to draw to.
		  */
		void setDrawBuffer (unsigned int number) const;

		/**
		  * Set multiple draw buffers to use.
		  * @param range from 0 to n where n is <= 15.
		  */
		void setMultipleDrawBuffers (unsigned int range) const;

		/**
		  * Show error.
		  */
		bool checkError (void) const;

		/**
		 * Returns whether the frame buffer object has been generated.
		 */
		bool valid (void) { return m_ids.size (); } const

		/**
		  * Returns the fbo id for this context.
		  */
		GLuint id (int context_id) const
		{
			return m_ids[context_id];
		}

	private:

		// Member variables.
		gfx::ContextBuffer <GLuint> m_ids;	// Buffer of instances of this class per context.
};

}

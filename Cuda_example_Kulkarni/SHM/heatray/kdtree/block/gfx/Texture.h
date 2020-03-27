/*
   Filename : Texture.h
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Purpose  : Encapsulate OpenGL texture calls. 

   Change List:

      - 06/18/2009  - Created (Cody White and Joe Mahsman)
*/

#pragma once

#include <GL/glew.h>
#include <gfx/ContextBuffer.h>
#include <math/Vector.h>
#include <util/string.h>
#include <FreeImage.h>
#include <iostream>

namespace gfx
{

/**
 * Encapsulates an simple 2D texture.
 */
class Texture
{
	public:

		/**
		  * Default constructor.
		  */
		Texture (void);

		/**
		  * Default destructor.
		  */
		~Texture (void);

		/**
		  * Destory a context.
		  * @param context_id ID of the context to destroy.
		  */
		void destroyContext (int context_id);

		struct Params;

		/**
		 * Sets the parameters used when creating the texture during load().
		 * See Texture::Params for a list of members and default values.
		 * @param p Parameters to set.
		 */
		void setParams (const Params& p) { m_params = p; }

		/**
		 * Load an image file using FreeImage and load it as an OpenGL texture.
		 * @param filename Path to the texture to load.
		 */
		void load (const char* filename);

		/**
		 * Creates the context-specific texture object and uploads the image data.
		 * @param context_id ID of the context to create the texture for.
		 */
		void initContext (int context_id);

		/**
		  * Create the context-specific texture object with passed-in image data.
		  * @param data Data to use for the texture.
		  * @param size Size of the texture.
		  * @param context_id ID of the context to use.
		  */
		void initContext (void *data, math::vec2i size, GLenum data_type, int context_id);

		/**
		 * Clears image data from core memory.
		 */
		void clearCPUData (void);
		
		/**
		  * Determine if this texture has any data loaded or not.
		  */
		bool hasData (void) const;

		/**
		  * Load a blank texture.
		  * @param width Width of the blank texture to create.
		  * @param height Height of the blank texture to create.
		  * @param data_type Type of texture to create (e.g. GL_RGBA).
		  */
		void blank (const size_t width, const size_t height, int data_type, int context_id);

		void bind (int active_texture = 0, int context_id = 0) const;
		void unbind (int active_texture = 0) const;

		/**
		 * Returns whether the texture object has been generated.
		 */
		bool valid (int context_id) const 
		{
			ContextBuffer<GLuint>::ConstContextBufferIterator iter = m_ids.find (context_id);
			if (iter != m_ids.end ())
			{
				return iter->second != 0;
			}

			return false;
		}

		/**
		  * Create mipmaps for this texture.
		  * WARNING: Only call this function if you used Texture::blank () to
		  * create the texture and wish for mipmaps to be created, otherwise
		  * they have alreay been created.
		  */
		void createMipmaps (int context_id);
		
		/**
		 * Returns the texture object ID assigned by OpenGL via glGenTextures().
		 */
		GLuint id (int context_id) { return m_ids[context_id]; }
		
		GLsizei width() const { return m_width; }
		GLsizei height() const { return m_height; }
		
		/**
		  * Get the pixel color at the passed in texture coordinates.
		  * _data must be valid for this function to work.
		  */
		math::Vector <float, 4> getPixelColor (float u, float v);

		/**
		  * Get a reference to the attachment value.
		  * @param context_id ID for the context.
		  */
		GLuint & attachment (int context_id);

		/**
		  * Get the attachment for this texture.
		  * @param context_id ID for this context.
		  */
		const GLuint attachment (int context_id) const;

		/**
		 * Contains parameters for texture operation.
		 * Pass this into a constructor or setParams().
		 */
		struct Params
		{
			/**
			 * number of color components
			 */
			GLint internalFormat;

			/**
			 * format of a color component
			 */
			GLenum format;

			/**
			 * Determines how s texture coordinates outside of [0..1]
			 * are handled.
			 */
			GLenum wrapS;

			GLenum wrapT;

			/**
			 * Defines the texturing function, used by OpenGL when a texture
			 * is applied to a primitive.
			 */
			GLint envMode;

			/**
			 * Minification filter used when a pixel contains
			 * multiple texels (e.g. the textured object is far).
			 */
			GLenum minFilter;

			/**
			 * Magnification filter used when a pixel is contained
			 * within a texel (e.g. the textured object is close).
			 * May only be GL_NEAREST or GL_LINEAR.
			 */
			GLenum magFilter;

			Params()
			{
				internalFormat = GL_RGB;
				format = GL_RGB;
				envMode = GL_MODULATE;
				wrapS = GL_REPEAT;
				wrapT = GL_REPEAT;	
				minFilter = GL_LINEAR_MIPMAP_LINEAR;
				magFilter = GL_LINEAR;
			}
		};

		util::string texture_name;	// Name of this texture.

	private:

		/**
		 * Set texture parameters through OpenGL API calls.
		 */
		void applyParams (void);

		// Member variables.
		gfx::ContextBuffer <GLuint> m_ids;  		// Context instances.
		gfx::ContextBuffer <GLuint> m_attachments; 	// Attachments to FBOs.
		GLsizei m_width;							// Width of the texture.
		GLsizei m_height;							// Height of the texture.

		Params m_params;							// Texture parameters.
	
		FIBITMAP* m_data;							// FreeImage texture data.

};
}


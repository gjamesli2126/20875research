/*
   Filename : VertexBuffer.h
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Change List:

      - 06/18/2009  - Created (Cody White and Joe Mahsman)
*/

#pragma once

#include <GL/glew.h>
#include <gfx/ContextBuffer.h>
#include <iostream>

namespace gfx
{

/**
 * Abstracts OpenGL vertex buffer objects.
 */
class VertexBuffer
{
	public:
		
		/** 
		 * Default constructor.
		 * @param target Type of VBO to use.
		 */
		VertexBuffer (GLenum target = GL_ARRAY_BUFFER, GLenum usage = GL_STATIC_DRAW)
		{
			m_target = target;
			m_usage = usage;
		}

		/** 
		 * Default destructor.
		 */
		~VertexBuffer (void)
		{
#ifdef USE_GL
			gfx::ContextBuffer<GLuint>::ContextBufferIterator iter;
			for (iter = m_ids.begin (); iter != m_ids.end (); ++iter)
			{
				glDeleteBuffers (1, &(iter->second));
			}

			m_ids.clear ();
#endif
		}

		/**
		  * Destroy a context.
		  * @param context_id ID of context to destrory.
		  */
		void destroyContext (int context_id)
		{
#ifdef USE_GL
			glDeleteBuffers (1, &m_ids[context_id]);
			m_ids.remove (context_id);
#endif
		}

		/**
		 * Assignment operator.
		 * @param other The vertex buffer to set this object equal to.
		 */
		VertexBuffer & operator= (const VertexBuffer &other)
		{
			if (this != &other)
			{
				gfx::ContextBuffer<GLuint>::ConstContextBufferIterator iter;
				for (iter = other.m_ids.begin (); iter != other.m_ids.end (); ++iter)
				{
					m_ids[iter->first] = iter->second;
				}

				m_target = other.m_target;
			}

			return *this;
		}

		/**
		 * Generate the VBO id and load the data.
		 * @param data The data to populate the VBO with.
		 * @param size Size in bytes of the data.
		 * @param context_id ID of the context to load data into.
		 */
		void load (void *data, int size, int context_id)
		{
#ifdef USE_GL
			GLuint& id = m_ids[context_id];
			glGenBuffers (1, &id);
			
			if (id == 0)
			{
				std::cout << "VertexBuffer::load () - Error: Unable to generate VBO!" << std::endl;
			}

			glBindBuffer (m_target, id);
			glBufferData (m_target, size, data, m_usage);
#endif
		}

		/**
		 * Bind the VBO for use. Must be called after
		 * load ().
		 */
		void bind (int context_id) const
		{
#ifdef USE_GL
			glBindBuffer (m_target, m_ids[context_id]);
#endif
		}

		/**
		 * Unbind this VBO from use.
		 */
		void unbind (void) const
		{
#ifdef USE_GL
			glBindBuffer (m_target, 0);
#endif
		}

		/** 
		 * Return the VBO id assigned by OpenGL. 
		 */
		GLuint id (int context_id) const
		{
			return m_ids[context_id];
		}

		/**
		  * Enable/Disable vertex arrays for a VBO.
		  * @param enable Enable/Disable.
		  */
		static void enableVertexArrays (bool enable)
		{
#ifdef USE_GL
			if (enable)
			{
				glEnableClientState (GL_VERTEX_ARRAY);
			}
			else
			{
				glDisableClientState (GL_VERTEX_ARRAY);
			}
#endif
		}

		/**
		  * Enable/Disable normal arrays for a VBO.
		  * @param enable Enable/Disable.
		  */
		static void enableNormalArrays (bool enable)
		{
#ifdef USE_GL
			if (enable)
			{
				glEnableClientState (GL_NORMAL_ARRAY);
			}
			else
			{
				glDisableClientState (GL_NORMAL_ARRAY);
			}
#endif
		}

		/**
		  * Enable/Disable tex coord arrays for a VBO.
		  * @param enable Enable/Disable.
		  */
		static void enableTexCoordArrays (bool enable)
		{
#ifdef USE_GL
			if (enable)
			{
				glEnableClientState (GL_TEXTURE_COORD_ARRAY);
			}
			else
			{
				glDisableClientState (GL_TEXTURE_COORD_ARRAY);
			}
#endif
		}

		/**
		  * Enable/Disable a vertex attribute array for this VBO.
		  * @param loc Location of the attribute variable.
		  * @param enable Enable/Disable
		  */
		static void enableVertexAttrib (GLint loc, bool enable)
		{
#ifdef USE_GL
			if (enable)
			{
				glEnableVertexAttribArrayARB (loc);
			}
			else
			{
				glDisableVertexAttribArrayARB (loc);
			}
#endif
		}

		/**
		  * Set the vertex data pointer for this vbo. 
		  * @param components Number of elements in a vertex.
		  * @param type Data type for this vbo.
		  * @param stride Stride.
		  * @param offset Start position for this data.
		  */
		void setVertexPointer (int components, GLenum type, int stride, int offset) const
		{
#ifdef USE_GL
			glVertexPointer (components, type, stride, (char *)NULL + offset);
#endif
		}

		/**
		  * Set the normal data pointer for this vbo. 
		  * @param type Data type for this vbo.
		  * @param stride Stride.
		  * @param offset Start position for this data.
		  */
		void setNormalPointer (GLenum type, int stride, int offset) const
		{
#ifdef USE_GL
			glNormalPointer (type, stride, (char *)NULL + offset);
#endif
		}

		/**
		  * Set the tex coord data pointer for this vbo. 
		  * @param components Number of elements in a tex coord.
		  * @param type Data type for this vbo.
		  * @param stride Stride.
		  * @param offset Start position for this data.
		  */
		void setTexCoordPointer (int components, GLenum type, int stride, int offset) const
		{
#ifdef USE_GL
			glTexCoordPointer (components, type, stride, (char *)NULL + offset);
#endif
		}

		/**
		  * Set the pointer for a vertex attribute.
		  * @param loc Location of the attribute.
		  * @param components Number of elements in the attribute.
		  * @param type Data type for this vbo.
		  * @param normalize If true, values in the array will be normalized in the range of -1 to 1 for signed data and 0 to 1 for unsigned data.
		  * @param stride Stride.
		  * @param offset Start position for this data.
		  */
		void setAttribPointer (GLint loc, int components, GLenum type, bool normalize, int stride, int offset) const
		{
#ifdef USE_GL
			glVertexAttribPointerARB (loc, components, type, normalize, stride, (char *)NULL + offset);
#endif
		}

		/**
		  * Render the VBO.
		  * @param type Type of primitive to render.  VBO must be bound.
		  * @param start Starting position in the array.
		  * @param size Number of elements in the VBO to render.
		  */
		void render (GLenum type, int start, int size) const
		{
#ifdef USE_GL
			glDrawArrays (type, start, size);
#endif
		}

	private:

		// Member variables.
		gfx::ContextBuffer <GLuint> m_ids;	// Per context instances.
		GLenum m_target;					// Target type.
		GLenum m_usage;						// How is this buffer to be used? (GL_STATIC_DRAW...).
};

}


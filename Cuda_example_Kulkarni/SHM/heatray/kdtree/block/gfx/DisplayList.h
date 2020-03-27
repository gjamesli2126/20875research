/*
   Filename : Display.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Create a globe for use in OpenGL that can act as a planet. 

   Change List:

      - 03/08/2010  - Created (Cody White)
*/

#pragma once

#include <GL/glew.h>
#include <gfx/ContextBuffer.h>
#include <gfx/util.h>
#include <iostream>

namespace gfx
{

class DisplayList
{
	public:

		/**
		  * Default constructor.
		  */
		DisplayList (void)
		{
		}

		/**
		  * Destructor.
		  */
		~DisplayList (void)
		{
			gfx::ContextBuffer<GLuint>::ContextBufferIterator iter;
			for (iter  = m_ids.begin ();
				 iter != m_ids.end ();
				 ++iter)
			{
				glDeleteLists (iter->second, 1);
			}

			m_ids.clear ();
		}

		/**
		  * Destroy context.
		  */
		void destroyContext (int context_id)
		{
			glDeleteLists (m_ids[context_id], 1);
			m_ids.remove (context_id);
		}

		/**
		  * Initialize a display list for use.
		  */
		void initialize (int context_id)
		{
			GLuint& id = m_ids[context_id];
			id = glGenLists (1);
			if (id == 0)
			{
				std::cout << "DisplayList::initialize () - Error: Unable to generate display list!" << std::endl;
				exit (0);
			}
		}

		/**
		  * Render the display list.
		  */
		void render (int context_id) const
		{
			glCallList (m_ids[context_id]);
		}

		/**
		  * Get the id for a given context.
		  */
		GLuint id (int context_id = 0) const
		{
			return m_ids[context_id];
		}

		/**
		  * Start the list.
		  */
		void startList (GLenum mode = GL_COMPILE, int context_id)
		{
			glNewList (m_ids[context_id], mode);
			checkGLErrors ("DisplayList::startList () - start a list");
		}
	
		/**
		  * End the list.
		  */
		void endList (void)
		{
			glEndList ();
		}

	private:

		gfx::ContextBuffer <GLuint> m_ids;
};

}



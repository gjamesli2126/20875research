/*
   Filename : Shader.h
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Purpose  : Encapsulates OpenGL shader functionality. 

   Change List:

      - 06/18/2009  - Created (Cody White and Joe Mahsman)
*/


#pragma once

#include <GL/glew.h>
#include <util/string.h>
#include <gfx/ContextBuffer.h>

namespace gfx
{

/**
 * Encapsulates the three GLSL shader types.
 */
class Shader
{
	public:

		/**
		  * Possible shader types.
		  */
		enum Type {	VERTEX, GEOMETRY, FRAGMENT };

		/**
		  * Default constructor.
		  */
		Shader (void);

		/**
		  * Destructor.
		  */
		~Shader (void);

		/**
		  * Delete a context.
		  * @param context_id Context to delete.
		  */
		void deleteContext (int context_id);

		/**
		 * Load the shader code from a text file and compile it
		 * as the specified shader type.
		 */
		void load (const char* filename);

		/**
		  * Initialize the shader for use.
		  */
		void initContext (Type type, int context_id);

		/**
		  * Delete a context.
		  * @param context_id ID of context to destroy.
		  */
		void destroyContext (int context_id);
		
		/**
		 * Returns the shader object ID assigned by OpenGL.
		 */
		GLuint id (int context_id) const { return m_ids[context_id]; } 

		/**
		  * Clear the CPU data from this shader.
		  */
		void clearCPUData (void);

	private:

		/**
		  * Read the shader file into memory.
		  * @param filename Path to the file to read.
		  */
		util::string readTextFile(const char *filename);

		/**
		  * Compile the shader for use.
		  * @param context_id ID for the context.
		  */
		void compile (int context_id);

		/**
		  * Create the shader.
		  * @param type Type of shader (VERTEX, FRAGMENT, or GEOMETRY).
		  * @param context_id ID for the context.
		  */
		void createShader (Type type, int context_id);

		// Member variables.
		gfx::ContextBuffer <GLuint> m_ids;	// Context-specific instances of the shaders.
		util::string m_source;				// Source from the input file.
		const char* m_filename;				// Filename to read.

};

}


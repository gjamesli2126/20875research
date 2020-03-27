/*
   Filename : Program.h
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Purpose  : Contains several shaders to makeup a shader program. 

   Change List:

      - 06/18/2009  - Created (Cody White and Joe Mahsman)
*/

#pragma once

#include <GL/glew.h>
#include <gfx/Shader.h>
#include <gfx/Uniform.h>
#include <gfx/ContextBuffer.h>
#include <util/string.h>
#include <map>

namespace gfx
{

/**
 * Encapsulates a GLSL program.
 */
class Program
{
	public:

		/**
		  * Default constructor.
		  */
		Program (void);

		/**
		  * Destructor.
		  */
		~Program (void);

		/**
		 * Attach the specified shader.
		 * The shader must already be loaded using Shader::load().
		 * @param shader Shader to attach to this program.
		 */
		void attach(Shader& shader, int context_id);

		/**
		 * Link the program.
		 * Call this after all shaders have been attached.
		 */
		void link (int context_id, const char *name = "Program");

		/**
		  * Destroy the context for this program.
		  */
		void destroyContext (int context_id);

		/**
		  * Initialize this program for geometry shader use.
		  * @param input OpenGL type of geometry for input.
		  * @param output OpenGL type of geometry for output.
		  * @param max_vertices The maximum number of vertices for the geometry shader to output.
		  * @param context_id ID of the context.
		  */
		void initGeometryParams (GLenum input, GLenum output, GLint max_vertices, int context_id);

		/**
		  * Add a shader directly to the program.
		  * @param filename Path to the shader code.
		  * @param type Type of shader.
		  * @param context_id ID of the context.
		  */
		void addShader (const char *filename, Shader::Type type, int context_id); 
 
		/**
		  * Set an integer uniform in the program.
		  * @param name Name of the uniform.
		  * @param i Value of the uniform.
		  */
		void set1i(const char* name, int i) const;

		/**
		  * Set a float uniform in the program.
		  * @param name Name of the uniform.
		  * @param f Value of the uniform.
		  */
		void set1f(const char* name, float f) const;

		/**
		  * Set 3 component double uniform in the program.
		  * @param name Name of the uniform.
		  * @param d Value of the uniform.
		  */
		void set3dv(const char* name, double *d) const;

		/**
		  * Set a 2 component float uniform in the program.
		  * @param name Name of the uniform.
		  * @param f Value of the uniform.
		  */
		void set2fv (const char *name, float *f) const;

		/**
		  * Set a 2 component uniform uniform in the program.
		  * @param name Name of the uniform.
		  * @param i Value of the uniform.
		  */
		void set2iv (const char *name, int *i) const;

		/**
		  * Set an 3 component float uniform in the program.
		  * @param name Name of the uniform.
		  * @param f Value of the uniform.
		  */
		void set3fv(const char* name, float *f) const;

		/**
		  * Set an 4 component uniform in the program.
		  * @param name Name of the uniform.
		  * @param d Value of the uniform.
		  */
		void set4iv(const char* name, int *d) const;

		/**
		  * Set a 4x4 matrix
		  * @param name Name of the uniform.
		  * @param d Value of the uniform.
		  */
		void setMatrix4fv(const char* name, float *f) const;

		/**
		  * Bind this program for use.
		  */
		void bind (int context_id) const;

		/**
		  * Unbind this program from use.
		  */
		void unbind() const;

		/**
		  * Get the id of this program.
		  */
		GLuint id (int context_id) const;

		/**
		  * Get a location of an attribute variable in the program.
		  * @param name Name of the attribute variable.
		  */
		GLint getAttributeLocation (const char *name, int context_id) const;

	private:

		gfx::ContextBuffer <GLuint> m_ids;

		mutable std::map<util::string, GLint> m_uniformCache;

		typedef std::map <util::string, GLint>::iterator CacheIterator;

		/**
		 * Iterate through all active uniforms and store their locations.
		 * Must be called after linking.
		 */
		void cacheUniforms (int context_id);
		void create (int context_id);

		util::string m_program_name;
};

}


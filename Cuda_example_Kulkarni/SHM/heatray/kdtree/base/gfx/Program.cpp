/*
   Filename : Program.cpp
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Purpose  : Contains several shaders to makeup a shader program. 

   Change List:

      - 06/18/2009  - Created (Cody White and Joe Mahsman)
*/

#include <gfx/Program.h>
#include <gfx/util.h>
#include <cstdlib>
#include <iostream>
using namespace std;

namespace gfx
{

// Default constructor.
Program::Program (void)
{
}

// Destructor.
Program::~Program (void)
{
#ifdef USE_GL
	gfx::ContextBuffer<GLuint>::ContextBufferIterator iter;
	for (iter = m_ids.begin (); iter != m_ids.end (); ++iter)
	{
		glDeleteProgram (iter->second);
	}

	m_ids.clear ();
#endif
}

// Attach a shader to the program.
void Program::attach (Shader& shader, int context_id)
{
#ifdef USE_GL
	create (context_id);
	glAttachShader (m_ids[context_id], shader.id (context_id));
	checkGLErrors("Program::attach () - Attach shader to program");
#endif
}

// Link the program.
void Program::link (int context_id, const char *name)
{
#ifdef USE_GL
	GLuint &id = m_ids[context_id];
	GLint success;
	const unsigned int bufferSize = 5000;
	char* buffer = new char[bufferSize];

	glLinkProgram(id);
	checkGLErrors("Program::link () - Link the program");

	glGetProgramiv(id, GL_LINK_STATUS, &success);
	checkGLErrors("Program::link () - Determine successful link");

	if (!success)
	{
		cout << "Problems with linking the program...\n"; 
		glGetProgramInfoLog(id, bufferSize, 0, buffer);
		printf("%s\n", buffer);

		delete [] buffer;
		exit(1);
	}

	m_program_name = name;
	cacheUniforms (context_id);
#endif
}

// Initialize geometry shader parameters.
void Program::initGeometryParams (GLenum input, GLenum output, GLint max_vertices, int context_id)
{
#ifdef USE_GL
	GLuint program_id = id (context_id);
	glProgramParameteriEXT (program_id, GL_GEOMETRY_INPUT_TYPE_EXT, input);
	glProgramParameteriEXT (program_id, GL_GEOMETRY_OUTPUT_TYPE_EXT, output);
	glProgramParameteriEXT (program_id, GL_GEOMETRY_VERTICES_OUT_EXT, max_vertices);
#endif
}

// Destroy this context.
void Program::destroyContext (int context_id)
{
#ifdef USE_GL
	glDeleteProgram (m_ids[context_id]);
	m_ids.remove (context_id);
#endif
}

void Program::set1i(const char* name, int i) const
{
#ifdef USE_GL
	CacheIterator iter = m_uniformCache.find (name);
	if (iter != m_uniformCache.end())
	{
		glUniform1i (iter->second, i);
	}
	else
	{
		//cout << "Program " << m_program_name << ": set1i error: " << name << endl;
	}
#endif
}

void Program::set1f(const char* name, float f) const
{
#ifdef USE_GL
	CacheIterator iter = m_uniformCache.find (name);
	if (iter != m_uniformCache.end())
	{
		glUniform1f (iter->second, f);
	}
	else
	{
		//cout << "Program " << m_program_name << ": set1f error: " << name << endl;
	}
#endif
}

void Program::set3dv(const char* name, double *d) const
{
#ifdef USE_GL
	CacheIterator iter = m_uniformCache.find (name);
	if (iter != m_uniformCache.end())
	{
		float f[3];
		f[0] = (float)d[0];
		f[1] = (float)d[1];
		f[2] = (float)d[2];
		
		glUniform3fv (iter->second, 1, f);
	}
	else
	{
		//cout << "Program " << m_program_name << ": set3dv error: " << name << endl;
	}
#endif
}

void Program::set2fv (const char *name, float *f) const
{
#ifdef USE_GL
	CacheIterator iter = m_uniformCache.find (name);
	if (iter != m_uniformCache.end ())
	{
		glUniform2fv (iter->second, 1, f);
	}
	else
	{
		//cout << "Program " << m_program_name << ": set2fv error: " << name << endl;
	}
#endif
}

void Program::set2iv (const char *name, int *i) const
{
#ifdef USE_GL
	CacheIterator iter = m_uniformCache.find (name);
	if (iter != m_uniformCache.end ())
	{
		glUniform2iv (iter->second, 1, i);
	}
	else
	{
		//cout << "Program " << m_program_name << ": set2fv error: " << name << endl;
	}
#endif
}

void Program::set3fv(const char* name, float *f) const
{
#ifdef USE_GL
	CacheIterator iter = m_uniformCache.find (name);
	if (iter != m_uniformCache.end())
	{
		glUniform3fv(iter->second, 1, f);
	}
	else
	{
		//cout << "Program " << m_program_name << ": set3fv error: " << name << endl;
	}
#endif
}

void Program::set4iv(const char* name, int *d) const
{
#ifdef USE_GL
	CacheIterator iter = m_uniformCache.find (name);
	if (iter != m_uniformCache.end())
	{
		glUniform4iv(iter->second, 1, d);
	}
	else
	{
		//cout << "Program " << m_program_name << ": set4iv error: " << name << endl;
	}
#endif
}

void Program::setMatrix4fv(const char* name, float *f) const
{
#ifdef USE_GL
	CacheIterator iter = m_uniformCache.find (name);
	if (iter != m_uniformCache.end())
	{
		glUniformMatrix4fv(iter->second, 1, false, f);
	}
	else
	{
		//cout << "Program " << m_program_name << ": setMatrix4fv error: " << name << endl;
	}
#endif
}

void Program::bind (int context_id) const
{
#ifdef USE_GL
	glUseProgram (m_ids[context_id]);
#endif
}

void Program::unbind (void) const
{
#ifdef USE_GL
	glUseProgram (0);
#endif
}

// Get the id of this program.
GLuint Program::id (int context_id) const
{
	return m_ids[context_id];
}

void Program::cacheUniforms (int context_id)
{
#ifdef USE_GL
	GLuint &id = m_ids[context_id];
	// Get number of active uniform variables
	GLint activeUniforms;
	glGetProgramiv (id, GL_ACTIVE_UNIFORMS, &activeUniforms);

	const int bufferSize = 200;
	char* nameBuffer = new char[bufferSize];

	GLsizei numChars;
	GLint size;
	GLenum dataType;

	// Get information on each
	for (int i = 0; i < activeUniforms; ++i)
	{
		glGetActiveUniform(id,
		                   i,
		                   bufferSize,
		                   &numChars,
		                   &size,
		                   &dataType,
		                   nameBuffer);

		char array[4];
		array[0] = nameBuffer[0];
		array[1] = nameBuffer[1];
		array[2] = nameBuffer[2];
		array[3] = 0;
		if (strcmp("gl_", array) == 0)
		{
			continue;
		}
		GLint location = glGetUniformLocation(id, nameBuffer);

		if (location == -1)
		{
			std::cout << "Program::cacheUniforms () - Error: Unable to cache uniform because a valid unifom location could not be found!" << std::endl;
			throw 0;
		}

		//TODO: This could be an issue -- it may need to be context safe
		m_uniformCache[nameBuffer] = location;
	}
#endif
}

void Program::create (int context_id)
{
#ifdef USE_GL
	GLuint &id = m_ids[context_id];
	if (glIsProgram (id))
	{
		return;
	}
	id = glCreateProgram();

	if (id == 0)
	{
		std::cout << "Program::create () - Error: OpenGL could not create the program!" << std::endl;
		throw 0;
	}
#endif
}

// Add a shader directly to the program.
void Program::addShader (const char *filename, Shader::Type type, int context_id)
{
#ifdef USE_GL
	gfx::Shader shader;
	shader.load (filename);
	shader.initContext (type, context_id);
	attach (shader, context_id);
#endif
}

// Get the location of an attribute variable in the program.
GLint Program::getAttributeLocation (const char *name, int context_id) const
{
#ifdef USE_GL
	return glGetAttribLocationARB (m_ids[context_id], name);
#else
	return 0;
#endif
}

}



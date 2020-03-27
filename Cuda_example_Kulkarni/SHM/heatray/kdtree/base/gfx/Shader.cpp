/*
   Filename : Shader.cpp
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Purpose  : Encapsulates OpenGL shader functionality. 

   Change List:

      - 06/18/2009  - Created (Cody White and Joe Mahsman)
*/

#include <gfx/Shader.h>
#include <gfx/util.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <fstream>

#define VERBOSE
using namespace std;

namespace gfx
{

// Default constructor.
Shader::Shader (void)
{
}

// Destructor.
Shader::~Shader (void)
{
#ifdef USE_GL
	gfx::ContextBuffer<GLuint>::ContextBufferIterator iter;
	for (iter = m_ids.begin (); iter != m_ids.end (); ++iter)
	{
		glDeleteShader (iter->first);
	}

	m_ids.clear ();
#endif
}

// Destroy this context.
void Shader::destroyContext (int context_id)
{
#ifdef USE_GL
	glDeleteShader (m_ids[context_id]);
	m_ids.remove (context_id);
#endif
}

// Read the shader file into memory.
util::string Shader::readTextFile (const char *filename)
{
	util::string content = "";
	std::ifstream fin;
	fin.open (filename);
	if (!fin)
	{
		return content;
	}

	std::string tmp;
	while (fin.good ())
	{
		tmp = fin.get ();
		if (fin.good ())
		{
			content += tmp;
		}
	}

	fin.close ();

	return content;
}

// Load the shader from the source file.
void Shader::load(const char* filename)
{
	m_filename = filename;
	m_source = readTextFile(filename);

	if (m_source == "")
	{
		std::cout << "Shader::load () - Error: could not read shader source file " << filename << std::endl;
		exit(1);
	}
}

// Initialize a shader for use.
void Shader::initContext (Type type, int context_id)
{
#ifdef USE_GL
	const GLchar *s = m_source.c_str ();
	createShader (type, context_id);
	glShaderSource(m_ids[context_id], 1, &s, NULL);
	checkGLErrors("Shader::initialize () - set shader source");
	compile (context_id);
	checkGLErrors("Shader::initialize () - compilation complete");
#endif
}

// Clear the CPU data from this shader.
void Shader::clearCPUData (void)
{
	m_source.clear ();
}

void Shader::compile (int context_id)
{
#ifdef USE_GL
	GLuint id = m_ids[context_id];
	GLint success;
	const unsigned int bufferSize = 5000;
	char* buffer = new char[bufferSize];
	memset (buffer, 0, bufferSize);

	glCompileShader(id);
	checkGLErrors("Shader::compile () - compiling shader");

	glGetShaderiv(id, GL_COMPILE_STATUS, &success);

	glGetShaderInfoLog(id, bufferSize, 0, buffer);
	if (buffer[0] != 0)
	{
		cout << "Problems with compiling " << m_filename << "..." << endl;
		printf("%s\n", buffer);
	}
	
	delete [] buffer;
#endif
}

// Create the shader.
void Shader::createShader(Type type, int context_id)
{
#ifdef USE_GL
	GLenum glType;

	switch (type)
	{
		case VERTEX:
			glType = GL_VERTEX_SHADER;
			break;

		case GEOMETRY:
			glType = GL_GEOMETRY_SHADER_EXT;
			break;

		case FRAGMENT:
			glType = GL_FRAGMENT_SHADER;
			break;

		default:
			std::cout << "Shader::createShader () - Error: Unknown type \"" << type << "\" specified as shader type. " 
					  << "Acceptable types are \"VERTEX\", \"GEOMETRY\", or \"FRAGMENT\"" << std::endl;
			exit(1);
	}

	GLuint &id = m_ids[context_id];
	checkGLErrors("Shader::createShader () - shader_type_set");
	id = glCreateShader(glType);
	checkGLErrors("Shader::createShader () - shader_id_created");

	if (id == 0)
	{
		std::cout << "Shader::createShader () - Error: Texture id == 0, shader could not be created!" << std::endl;
		exit(1);
	}
#endif
}

}


/*
   Filename : Skybox.h
   Author   : Joe Mahsman
   Version  : 1.0

   Purpose  : Implements a easy-to-use interface to create a skybox around the camera.

   Change List:

      - 06/12/2009  - Created (Joe Mahsman)
*/

#pragma once

#include <GL/glew.h>
#include <util/string.h>

#include <gfx/Texture.h>
#include <gfx/ContextBuffer.h>
#include <math/Vector.h>

namespace gfx
{

class Skybox
{
	public:

		/**
		  * Default constructor.
		  */
		Skybox (void);

		/** 
		  * Destructor.
		  */
		virtual ~Skybox (void);
		
		/** 
		 * Initialize the skybox with a directory path (relative or absolute). 
		 */
		void load(const util::string& dir);

		/**
		 * Create the new graphics context.
		 */
		void initContext(int context_id);

		/**
		 * Destroy the created graphics context.
		 */
		void destroyContext(int context_id);

		/**
		 * Render the skybox.  This can only be called after initializing the class.
		 */
		void render(int context_id);

		/**
		  * Set the position of the skybox.
		  * This should be the eye position.
		  */
		void setPosition (const math::vec3f& position, int id);
		void setPosition (const float* position, int id);

		/**
		 * Set the amount by which to scale the box that is
		 * rendered around the user. The default is 1.0f.
		 */
		void setScale(float scale) { _scale = scale; }

	private:
	
		/** 
		  * Checks whether all six files with an extension exist in the directory
		  * provided. 
		  */
		bool filesExist (const util::string& dir, const util::string& type);

		// The skybox images are organized as follows:
		// 0 : looking down the positive x axis
		// 1 : negative x
		// 2 : positive y
		// 3 : negative y
		// 4 : positive z
		// 5 : negative z
		gfx::Texture _images[6];

		gfx::ContextBuffer<math::vec3f> _positions;
		//math::vec3f _position;
		float _scale;
};

}

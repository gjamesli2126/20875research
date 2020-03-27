/*

   Filename : World.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Mediator between OpenGL and ray tracer. 

   Change List:

      - 12/20/2009  - Created (Cody White)

*/

#pragma once

#include "gfx/gfx.h"

#include <gfx/Camera.h>
#include <gfx/Mesh.h>
#include <math/Vector.h>
#include <math/Constants.h>
#include <kdtree/TriangleKDTree.h>
#include <raytracer/Raytracer.h>
#include <shared/Light.h>
#include <vector>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <fstream>
#include "raytracer/Params.h"

class World
{
	public:
		
		/**
		  * Default constructor.
		  */
		World (void)
		{
			CAMERA_SPEED = 6.5f;
			ROTATE_SPEED = 1.6f;
			m_raytracing = false;
		}

		/**
		  * Default destructor.
		  */
		~World (void)
		{
		}

		void init (int argc, char **argv);
		void printHelp (void);
		void update (const float dt);
		void render(void);
		void setProjection (int width, int height, float fov, float z_near, float z_far);
		void rotateCamera (int yaw, int pitch);
		void start();
		
		bool normal_keys [256];
		bool special_keys [256];

		int getScreenWidth() { return m_screen_width; }
		int getScreenHeight() { return m_screen_height; }

	private:

		void checkKeys (const float dt);
		void placeLights();
		void configure (int argc, char **argv);
		
		// Member variables.
		gfx::Camera m_camera;	// Camera for viewing in OpenGL.
		gfx::Mesh	m_mesh;		// Triangle mesh for ray tracing.
		
		int m_screen_width;		// Width of the screen in pixels.
		int m_screen_height;	// Height of the screen in pixels.

		float CAMERA_SPEED;		// Speed at which the camera moves.
		float ROTATE_SPEED;		// Speed at which the camera rotates.

		bool m_raytracing;		// Flag to determine if the code is currently ray tracing or not.

		Raytracer m_raytracer;	// Raytracer class to perform the actual ray tracing.
		TriangleKDTree m_tree;	// KD-Tree which stores the subdivided triangle mesh.

		std::vector <Light > m_lights;	// All lights in the scene.

		math::Vector <float, 3> m_light_power;	// Power for the lights in the scene.

		math::vec2f m_camera_vector;	// Vector for moving the camera with the mouse.
};

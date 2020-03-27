/*
 * World.cpp
 *
 *  Created on: Oct 30, 2012
 *      Author: yjo
 */

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

#include "raytracer/World.h"

#define GLUT_KEY_ENTER		13
#define GLUT_KEY_ESCAPE		27
#define GLUT_KEY_SPACE      32

#ifdef METRICS
int numberOfTraversals=0;
int splice_depth;
int max_depth=0;
std::vector<TriangleKDTreeNode*> subtrees;
std::vector<subtreeStats> subtreeStatsList;
#endif

/**
 * Initialize this class for use.
 */
void World::init (int argc, char **argv)
{
	m_screen_width  = 0;
	m_screen_height = 0;

	memset (normal_keys, 0, 256);
	memset (special_keys, 0, 256);

	configure (argc, argv);

	setProjection (m_screen_width, m_screen_height, 45.0f, 0.1f, 10000.0f);
}

/**
 * Print command line help.
 */
void World::printHelp (void)
{
	std::cout << "Usage: " << std::endl;
	std::cout << "\t-i [path to .obj model] -a [anti-aliasing level]" << std::endl;

	std::cout << std::endl << "WASD moves the camera, R and F pivot the camera up and down, Q and E left and right." << std::endl;
	std::cout << "L adds a light to the scene at the current camera position." << std::endl;
	std::cout << "K removes the most recently placed light" << std::endl;
	std::cout << "Press enter to start the ray tracing process." << std::endl << std::endl;
}

/**
 * Update this class.
 */
void World::update (const float dt)
{
	checkKeys (dt);

	if (m_camera_vector != math::vec2f::zero ())
	{
		m_camera.yaw (m_camera_vector[0] * dt * ROTATE_SPEED);
		m_camera.pitch (m_camera_vector[1] * dt * ROTATE_SPEED);
		m_camera_vector = math::vec2f::zero ();
	}
}

/**
 * Render the screen.
 */
void World::render(void)
{
#ifdef USE_GL
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity ();

	m_camera.apply ();

	if (!m_raytracing)
	{
		//glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);

		math::vec4f light_pos = m_camera.getPosition ();
		light_pos[3] = 1.0f;
		glLightfv (GL_LIGHT0, GL_POSITION, light_pos.v);

		m_mesh.render (0);
		//m_tree.render ();
		//m_tree.renderBorders ();

		GLUquadric *sphere = gluNewQuadric ();
		// Render the lights.
		glDisable (GL_LIGHTING);
		for (size_t i = 0; i < m_lights.size (); ++i)
		{
			glColor3f (1.0f, 1.0f, 1.0f);
			glPushMatrix ();
			glTranslatef (m_lights[i].position[0], m_lights[i].position[1], m_lights[i].position[2]);
			gluSphere (sphere, 0.1f, 10, 10);
			glPopMatrix ();

			glBegin (GL_LINES);
			glColor3f (1.0f, 0.0f, 0.0f);
			glVertex3fv (m_lights[i].position.v);
			glVertex3fv ((m_lights[i].position + m_lights[i].forward).v);

			glColor3f (0.0f, 1.0f, 0.0f);
			glVertex3fv (m_lights[i].position.v);
			glVertex3fv ((m_lights[i].position + m_lights[i].up).v);

			glColor3f (0.0f, 0.0f, 1.0f);
			glVertex3fv (m_lights[i].position.v);
			glVertex3fv ((m_lights[i].position + m_lights[i].right).v);

			glEnd ();
		}
		glEnable (GL_LIGHTING);

		gluDeleteQuadric (sphere);
	}

	else
	{
		m_raytracer.render ();
	}

	glutSwapBuffers();
#endif
}

/**
 * Set projection.
 */
void World::setProjection (int width, int height, float fov, float z_near, float z_far)
{
	m_screen_width  = width;
	m_screen_height = height;

	m_camera.setPerspective (fov, (float)((float)width / (float)height), z_near, z_far);

	// Loop through the lights and reset their projection information.
	for (size_t i = 0; i < m_lights.size (); ++i)
	{
		m_lights[i].fov_x = m_camera.getFOVX () * math::DEGREE_TO_RADIAN;
		m_lights[i].fov_y = m_camera.getFOVY () * math::DEGREE_TO_RADIAN;
	}
}

void World::rotateCamera (int yaw, int pitch)
{
	m_camera_vector = math::vec2f (yaw, pitch);
}

void World::start() {
	//m_camera.output();
	m_raytracing = true;
	m_raytracer.start (m_screen_width, m_screen_height, m_camera.getForwardVector(), m_camera.getUpVector(), m_camera.getRightVector(), m_camera.getPosition(), &m_lights);
}

void World::checkKeys (const float dt)
{
	static float delta = 0;
	delta -= dt;
	if (normal_keys[GLUT_KEY_ESCAPE])
	{
		exit (0);
	}

	if (normal_keys['w'])
	{
		m_camera.move (0.0f, 0.0f, -CAMERA_SPEED * dt);

	}

	if (normal_keys['s'])
	{
		m_camera.move (0.0f, 0.0f, CAMERA_SPEED * dt);

	}

	if (normal_keys['a'])
	{
		m_camera.move (-CAMERA_SPEED * dt, 0.0f, 0.0f);

	}

	if (normal_keys['d'])
	{
		m_camera.move (CAMERA_SPEED * dt, 0.0f, 0.0f);

	}

	if (normal_keys['r'])
	{
		m_camera.pitch (ROTATE_SPEED * dt);

	}

	if (normal_keys['f'])
	{
		m_camera.pitch (-ROTATE_SPEED * dt);

	}

	if (normal_keys['e'])
	{
		m_camera.yaw (-ROTATE_SPEED * dt);

	}

	if (normal_keys['q'])
	{
		m_camera.yaw (ROTATE_SPEED * dt);

	}

	if (normal_keys['z'])
	{
		m_camera.roll (0.6f * dt);

	}

	if (normal_keys['c'])
	{
		m_camera.roll (-0.6f * dt);

	}

	if (normal_keys['l'] && delta <= 0.0f)
	{
		delta = 0.5f;
		// Place a light in the scene.
		Light light;
		light.position = m_camera.getPosition ();
		light.forward  = m_camera.getForwardVector ();
		light.up	   = m_camera.getUpVector ();
		light.right	   = m_camera.getRightVector ();
		light.fov_y	   = m_camera.getFOVY () * math::DEGREE_TO_RADIAN;
		light.fov_x    = m_camera.getFOVX () * math::DEGREE_TO_RADIAN;
		//light.fov_y    = 90.0f * math::DEGREE_TO_RADIAN;
		//light.fov_x    = 90.0f * math::DEGREE_TO_RADIAN;
		light.power	   = m_light_power;
		light.output();
		m_lights.push_back (light);
	}

	if (normal_keys['k'] && delta <= 0.0f)
	{
		if (m_lights.size ())
		{
			delta = 0.5f;
			// Remove the last light created.
			m_lights.erase (--m_lights.end ());
		}
	}

	if (normal_keys[GLUT_KEY_ENTER] && delta <= 0.0f)
	{
		m_camera.output();
		delta = 0.5f;
		// Start/stop ray tracing.
		//m_raytracing = !m_raytracing;
		m_raytracing = true;

		if (m_raytracing == true)
		{
			m_raytracer.start (m_screen_width, m_screen_height, m_camera.getForwardVector (), m_camera.getUpVector (), m_camera.getRightVector (), m_camera.getPosition (), &m_lights);
		}
	}
}

void World::placeLights() {
	math::vec3f forward(0.0f, 0.0f, -1.0f);
	math::vec3f up(0.0f, 1.0f, 0.0f);
	math::vec3f right(1.0f, 0.0f, 0.0f);
	float fov_x, fov_y;
	fov_x = fov_y = 0.7854f;
	vector<math::vec3f> pos;
	pos.push_back(math::vec3f(5.0f, 5.0f, 50.0f));
	pos.push_back(math::vec3f(-5.0f, 5.0f, 50.0f));
	pos.push_back(math::vec3f(-5.0f, -5.0f, 50.0f));
	pos.push_back(math::vec3f(5.0f, -5.0f, 50.0f));
	pos.push_back(math::vec3f(5.0f, 0.0f, 50.0f));
	pos.push_back(math::vec3f(-5.0f, 0.0f, 50.0f));
	pos.push_back(math::vec3f(0.0f, 5.0f, 50.0f));
	pos.push_back(math::vec3f(0.0f, -5.0f, 50.0f));
	pos.push_back(math::vec3f(0.0f, 0.0f, 50.0f));

	for (int i = 0; i < pos.size(); i++) {
		Light light;
		light.position = pos[i];
		light.forward  = forward;
		light.up	   = up;
		light.right	   = right;
		light.fov_y	   = fov_y;
		light.fov_x    = fov_x;
		light.power	   = m_light_power;
		m_lights.push_back (light);
	}
}


/**
 * Configure the ray tracer based on the config file "heatray.config".
 */
void World::configure (int argc, char **argv)
{
	int anti_alias = 1;
	util::string model_file = "";
	size_t photons_per_light = 0;
	size_t num_gather = 0;
	size_t max_gather_distance = 4;
	size_t caustic_density = 3;

	model_file = util::string(argv[ModelFile]);
	photons_per_light = atoi(argv[PhotonsPerLight]);
	m_light_power[0] = m_light_power[1] = m_light_power[2] = atoi(argv[LightPower]);
	m_screen_width = atoi(argv[ScreenWidth]);
	m_screen_height = atoi(argv[ScreenHeight]);
	//m_camera.setPosition (math::Vector <float, 3> ((float)0, (float)5, (float)15));
	num_gather = 350;

	//placeLights();
	string light_file = model_file.substr(0, model_file.length() - 3) + "txt";
	cout << light_file << endl;
	std::ifstream fin;
	fin.open(light_file.c_str());
	util::string type;
	while (fin >> type) {
		if (type == "light") {
			Light light;
			light.load(fin, m_light_power);
			//light.output();
			m_lights.push_back (light);
		} else if (type == "camera") {
			m_camera.load(fin);
			//m_camera.output();
		}
	}

	if (!m_mesh.load (model_file, true)) {
		exit (0);
	}

	// Loop through the materials and set the transparent to the ambient
	// if this is an obj file.
	if (model_file.find (".obj"))
	{
		for (size_t i = 0; i < m_mesh.getNumMeshes (); ++i)
		{
			gfx::Mesh::MeshPiece *piece = m_mesh.getMesh (i);
			piece->material.transmissive = piece->material.ambient;
		}
	}

	cout << "num meshes: " << m_mesh.getNumMeshes() << endl;
	cout << "num triangles: " << m_mesh.getNumTriangles() << endl;
#ifdef METRICS
	splice_depth = 13; //hardcoded values for christmas.
#endif
	// Load the data into the tree.
	m_tree.build (m_mesh);

	// Initialize the raytracer for use.
	m_raytracer.initialize (&m_tree, photons_per_light, num_gather, max_gather_distance, caustic_density);
}


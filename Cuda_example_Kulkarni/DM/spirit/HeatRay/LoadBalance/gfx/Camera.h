/*

   Filename : Camera.h
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Purpose  : Implements a quaternion based camera for use with OpenGL. 

   Change List:

      - 06/12/2009  - Created (Cody White and Joe Mahsman)

	  - 10/14/2009	- Added field-of-view calculations into this class (Cody White)
*/

#pragma once

#include <GL/glew.h>
#include <gfx/util.h>
#include <math/Vector.h>
#include <math/Quaternion.h>
#include <math/Matrix.h>
#include <fstream>

namespace gfx
{

/**
  * OpenGL-based camera that uses quaternions to represent rotations.
  */
class Camera
{

	public:

		/**
		  * Default constructor.
		  */
		Camera (void);

		/** 
		  * Destructor.
		  */
		virtual ~Camera (void);

		/** 
		  * Apply transformations to the modelview matrix. 
		  */
		virtual void apply (void) const;

		/** 
		  * Set the position of the camera.
		  * @param position Position to get the camera to.
		  */		
		void setPosition (const math::vec3f &position);

		/**
		  * Get the current position of the camera.  Returns a math::vec3f.
		  */
		const math::vec3f getPosition (void) const;

		/** 
		  * Move the camera along it local axis.
		  * @param x_distance Distance to move the camera in the x direction.
		  * @param y_distance Distance to move the camera in the y direction.
		  * @param z_distance Distance to move the camera in the z direction.
		  */
		void move (float x_distance, float y_distance, float z_distance);

		/**
		 * Move the camera along its forward vector.
		 */
		void move (float negz_distance);

		/**
		 * Move the camera along its right vector (x-axis).
		 */
		void strafe (float x_distance);

		/**
		  * Rotate about the x axis. Value is in radians.
		  * @param angle Angle of rotation about the x axis.
		  */
		void pitch (float angle);

		/**
		  * Rotate about the y axis. Value is in radians.
		  * @param angle Angle of rotation about the y axis.
		  */
		void yaw (float angle);

		/**
		  * Rotate about the z axis. value is in radians.
		  * @param angle Angle of rotation about the z axis.
		  */
		void roll (float angle);

		/**
		  * Rotate about the world y axis. value is in radians.
		  * This is useful for FPS-style navigation.
		  * @param angle Angle of rotation about the z axis.
		  */
		void turn (float angle);

		/** 
		  * Rotate about an arbitrary world-space axis (Can contain mutliple rotations).
		  * @param q Quaternion to apply to the current orientation that rotates about the fixed-world axis.
		  */
		void rotateWorld (math::quatf q);

		/**
		  * Rotate about an arbitrary object-space axis (Can contain mutliple rotations).
		  * @param q Quaternion to apply to the current orientation that rotates about the orientation's local axis.
		  */
		void rotateLocal (math::quatf q);

		/**
		  * Render the coordinate axis of the camera for debugging purposes.
		  */
		virtual void debugRender (void) const;

		/**
		  * Get the normalized forward vector from the current view.
		  */
		const math::vec3f getForwardVector (void) const;

		/**
		  * Get the normalized up vector from the current view.
		  */
		const math::vec3f getUpVector (void) const;

		/**
		  * Get the normalized right vector from the current view.
		  */
		const math::vec3f getRightVector (void) const;

		/**
		  * Set the current view perspective of the camera.
		  * @param fov Field of view of the camera.
		  * @param aspect_ratio Aspect ratio of the viewport.
		  * @param z_near Near plane.
		  * @param z_far Far place.
		  */
		void setPerspective (float fov, float aspect_ratio, float z_near, float z_far);

		/**
		  * Get the current model view matrix.
		  */
		math::Mat4f getModelViewMatrix (void) const;

		/**
		  * Get the current projection matrix.
		  */
		math::Mat4f getProjectionMatrix (void) const;

		/**
		  * Get the aspect ratio of the camera.
		  */
		float getAspectRatio (void) const;

		/**
		  * Get the field of view of the camera (x).
		  */
		float getFOVX (void) const;

		/**
		  * Get the field of view of the camera (y).
		  */
		float getFOVY (void) const;

		void output() {
			printf("camera %f %f %f %f %f %f\n", m_position.v[0], m_position.v[1], m_position.v[2], m_aspect_ratio, m_fov_y, m_fov_x);
			m_orientation.output();
			m_view_matrix.output();
			m_projection_matrix.output();
		}

		void load(std::ifstream& fin) {
			fin >> m_position.v[0] >> m_position.v[1] >> m_position.v[2] >> m_aspect_ratio >> m_fov_y >> m_fov_x;
			m_orientation.load(fin);
			m_view_matrix.load(fin);
			m_projection_matrix.load(fin);
		}

	protected:

		/**
		  * Update the internally stored view matrix.
		  */
		virtual void updateViewMatrix (void);

		/**
		  * Update position of the camera.
		  */
		void updateMatrixPosition (void);

		// Member variables.
		math::vec3f	m_position;				// Current position of the camera.
		math::quatf m_orientation;			// Current orientation of the camera.
		math::Mat4f m_view_matrix;			// Stored modelview matrix for the current orientation of the camera.
		math::Mat4f	m_projection_matrix;	// Stored projection matrix for the current persepctive of the camera.
		float		m_aspect_ratio;			// Aspect ratio of the camera.
		float       m_fov_y;				// Field of view of the camera (y).
		float       m_fov_x;				// Field of view of the camera (x).
};

}



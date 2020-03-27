/*

   Filename : ThirdPersonCamera.h
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Change List:

      - 06/12/2009  - Created (Cody White and Joe Mahsman)

	  - 06/25/2009  - Added fromAxisAngle function (Cody white)
*/

#pragma once

#include <GL/glew.h>
#include <iostream>
#include <gfx/Camera.h>

namespace gfx
{

/**
 * This camera orbits about a target point.  Inherits from Camera.
 */
class ThirdPersonCamera : public Camera
{
	public:
		
		/**
		  * Default constructor.
		  * @param orbit_point Point to orbit the camera about.
		  * @param orbit_distance Initial distance of the camera from the orbit point.
		  * @param continuous_orbit Enable continuous orbiting of the camera.
		  * @param continuous_pivot Enable continuous pivoting of the camera.
		  */
		ThirdPersonCamera (math::vec3f orbit_point = math::vec3f (0.0f, 0.0f, 0.0f),
						   float orbit_distance = 0.0f,
						   bool continuous_orbit = false,
						   bool continuous_pivot = false,
						   bool scale_orbit = false) : Camera ()
		{
			m_zoom_time  	 	= 0.0f;
			m_zoom_dir   	 	= 0;
			m_orbiting   	 	= false;
			m_pivoting   	 	= false;
			m_orbit_point    	= orbit_point;
			m_orbit_distance 	= orbit_distance;
			ZOOM_SPEED		 	= 50.0f;
			PIVOT_SPEED		 	= 0.05f;
			m_min_distance 	 	= HUGE_VAL;
			m_max_distance   	= HUGE_VAL;
			m_scale_zoom	 	= false;
			m_continuous_orbit 	= continuous_orbit;
			m_continuous_pivot 	= continuous_pivot;
			m_scale_orbit		= scale_orbit;
			m_min_orbit_speed	= 0.0f;
			m_max_orbit_speed	= HUGE_VAL;

			m_look_at.identity ();
		}

		/**
		  * Default destructor.
		  */
		~ThirdPersonCamera (void)
		{
		}

		/** 
		  * Set the minimum distance the camera can be from the lookat point. 
		  * @param distance Minimum orbit distance for the camera.
		  */
		void setMinOrbit (float distance)
		{
			m_min_distance = distance;
		}

		/** 
		  * Set the maximum orbit distance the camera can be from the orbit point. 
		  * @param distance Maximum orbit distance for the camera.
		  */
		void setMaxOrbit (float distance)
		{
			m_max_distance = distance;
		}

		/**
		  * Enable/disable continuous orbiting.
		  * @param enable Flag to enable/disable continuous orbiting.
		  */
		void enableContinuousOrbit (bool enable)
		{
			m_continuous_orbit = enable;
		}

		/**
		  * Enable/disable continuous pivoting.
		  * @param Flag to enable/disable continuous pivoting.
		  */
		void enableContinuousPivoting (bool enable)
		{
			m_continuous_pivot = enable;
		}

		/**
		  * Enable/disable scaling the orbit speed.
		  * @param Flag to enable/disable orbit speed scaling.
		  */
		void enableOrbitSpeedScaling (bool enable)
		{
			m_scale_orbit = enable;
		}

		/**
		  * Set the minimum speed the camera can move at.
		  * @param speed Minimum speed of the camera.
		  */
		void setMinimumOrbitSpeed (float speed)
		{
			m_min_orbit_speed = speed;
		}

		/**
		  * Set the maximum orbit speed of the camera.
		  * @param speed Maximum orbit speed of the camera.
		  */
		void setMaximumOrbitSpeed (float speed)
		{
			m_max_orbit_speed = speed;
		}

		/** Use zoom scaling.  This changes the zoom speed based on the distance of the
		  *	camera from the min distance.  NOTE: the min distance must be set before 
		  *	this function can have any effect. 
		  * @param use Flag to enable/disable zoom scaling.
		  */
		void useZoomScaling (bool use)
		{
			m_scale_zoom = use;
		}

		/** 
		  * Get the orbit distance. 
		  */
		float getOrbitDistance (void) const
		{
			return m_orbit_distance;
		}

		/** 
		  * Set the orbit distance.  If this distance is outside of the bounds of the current
		  * minimum and maximum orbit distances, it will be clamped.
		  * @param distance Distance of the camera from the orbit point.
		  */
		void setOrbitDistance (float distance)
		{
			m_orbit_distance = distance;
		}

		/** 
		  * Tell the camera to zoom in or out.
		  * @param in Zoom in == true, zoom out == false.
		  */
		void zoom (bool in, float dt)
		{
			if (in)
			{
				if (m_zoom_dir == -1)
				{
					m_zoom_time = dt;
				}

				m_zoom_dir = 1;
				m_zoom_time += dt;
			}
			else
			{
				if (m_zoom_dir == 1)
				{
					m_zoom_time = dt;
				}

				m_zoom_dir = -1;
				m_zoom_time += dt;
			}
		}

		/** 
		  * Initiate orbit motion. 
		  * @param vector 2D vector to orbit with (latitude, longitude).
		  */
		void orbit (const math::vec2f &vector)
		{
			m_orbit_vector = vector;
			m_orbiting = true;

			if (m_scale_orbit)
			{
				float speed = (((math::length (m_position - m_orbit_point)) - m_min_distance)) / (m_max_distance - m_min_distance);
				if (speed > 0.0f)
				{
					// Perform an exponential scaling of the speed instead of linear.  The farther away, the faster the
					// camera will move.
					//speed = powf (speed, 1.5f);
				}

				speed = std::max (m_min_orbit_speed, std::min (m_max_orbit_speed, speed));
				m_orbit_vector = m_orbit_vector * speed;
			}
		}

		/** 
		  * Initiate pivoting.
		  * @param vector 2D vector to pivot with (latitue, longitude).
		  */
		void pivot (const math::vec2f &vector)
		{
			m_pivot_vector = vector;
			m_pivoting = true;
		}

		/**
		  * Immediately stop orbiting the camera around the orbit point.
		  */
		void stopOrbit (void)
		{
			m_orbiting = false;
		}

		/** 
		  * Render the camera as an object in the world. 
		  */
		virtual void renderDebug ()
		{
			glDisable(GL_LIGHTING);
			glPushMatrix();

			m_position = m_orbit_point + m_orientation.getZAxis () * m_orbit_distance;

			math::Matrixf R;
			math::quatf tmp = m_orientation;
			tmp.toMatrix (R);

			glMultMatrixf (R.v);
			glScaled (5000, 5000, 5000);

			math::vec3f forward (0, 0, -1);
			math::vec3f up (0, 1, 0);
			math::vec3f right (1, 0, 0);

			glBegin(GL_LINES);
				glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3fv(forward.v);
				glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3fv(up.v);
				glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3fv(right.v);
			glEnd();
			glPopMatrix();
			glEnable(GL_LIGHTING);
		}

		/** 
		  * Set the target (look at point) of the camera. 
		  * @param target Point to look at in world space.
		  */
		void lookAt (const math::vec3f &target)
		{
			math::vec3f new_look_at = normalize (target - m_position);
			math::vec3f current_look_at = normalize (getForwardVector () - m_position);

			if (new_look_at == current_look_at) // There is nothing to change.
			{
				return;
			}

			float angle = dot (current_look_at, new_look_at);
			math::vec3f normal = cross (current_look_at, new_look_at);

			// Create the new look at quaternion.
			m_look_at = math::quatf (angle, normal, true);
			m_look_at.normalize ();
			updateViewMatrix ();
		}

		/** 
		  * Set the orbit point for the camera to orbit about. 
		  * @param orbit Point to orbit the camera about.
		  */
		void setOrbitPoint (const math::vec3f &orbit)
		{
			m_orbit_point = orbit;
		}

		/** 
		  * Update the camera's position and orienation; this should be called every frame. 
		  * @param dt Time since the last update call.
		  */
		void update(float dt)
		{
			checkZoom (dt);
			checkOrbit (dt);
			checkPivot (dt);

			// Update the position of the camera.
			m_position = m_orbit_point + m_orientation.getZAxis () * m_orbit_distance;
			updateMatrixPosition ();
		}

		/** 
		  * Immediately stop pivoting the camera. 
		  */
		void stopPivot (void)
		{
			m_pivoting = false;
		}

	protected:

		/** 
		  * Update the internally stored view matrix. 
		  */
		virtual void updateViewMatrix (void)
		{
			math::quatf tmp = m_look_at * m_orientation;
			tmp.toMatrix (m_view_matrix);
			updateMatrixPosition ();
		}

		/** 
		  * Update the zoom value if applicable. 
		  * @param dt Time since last update.
		  */
		void checkZoom (float dt)
		{
			if (m_zoom_time > 0.0f)
			{
				m_zoom_time -= dt;

				if (m_zoom_time < 0.0f)
				{
					m_zoom_time = 0.0f;
				}

				// If the zoom speed should be scaled, scale it based on the camera distance.
				if (m_scale_zoom)
				{
					float distance = m_orbit_distance - m_min_distance;
					ZOOM_SPEED = distance;	
				}

				// Zoom in/out
				m_orbit_distance += -(ZOOM_SPEED * dt * m_zoom_dir);

				// Force the orbit distance to be greater than the minimum.
				if (m_orbit_distance < m_min_distance)
				{
					m_orbit_distance = m_min_distance;
				}
				else if (m_orbit_distance > m_max_distance)
				{
					m_orbit_distance = m_max_distance;
				}
			}
		}

		/**
		  * Update the position if applicable. 
		  * @param dt Time since last update.
		  */
		void checkOrbit (float dt)
		{
			if (m_orbiting)
			{
				math::quatf q1, q2;
				q1.createFromAxisAngle (-m_orbit_vector[1] * dt, m_look_at.getXAxis ());
				q2.createFromAxisAngle (-m_orbit_vector[0] * dt, m_look_at.getYAxis ());
				rotateWorld (math::normalize (q1 * q2));
				m_orbiting = m_continuous_orbit;
			}
		}

		/** 
		  * Update the current rotation of the camera if applicable. 
		  * @param dt Time since last update.
		  */
		void checkPivot (float dt)
		{
			if (m_pivoting)
			{
				math::quatf tmp;
				math::Quaternion<float>::fromYawPitchRoll (m_pivot_vector[0] * PIVOT_SPEED * dt,
								  			  			   m_pivot_vector[1] * PIVOT_SPEED * dt,
								  			  			   0.0f, 
								 			  			   tmp);
				m_look_at = math::normalize (tmp * m_look_at);
				updateViewMatrix ();
				m_pivoting = m_continuous_pivot;
			}
		}

		// Camera speeds
		float ZOOM_SPEED;
		float PIVOT_SPEED;

		int 	m_zoom_dir;			// Direction of the camera to go in the next zoom update.
		float 	m_zoom_time;		// Time left to keep zooming the camera.
		bool 	m_orbiting;			// Flag to determine if the camera should be orbiting or not.
		bool 	m_pivoting;			// Flag to determine if the camera should be pivoting or not.
		bool	m_scale_zoom;		// Flag to determine if the camera should scale the zoom speed based upon the distance of the camera.
		bool 	m_continuous_orbit;	// Flag to tell if the camera should continuously orbit.
		bool	m_continuous_pivot;	// Flag to tell if the camera should continuously pivot.
		bool	m_scale_orbit;		// Flag to tell if the camera should scale the speed of the orbit.

		math::vec2f m_orbit_vector;	// Vector to orbit the camera about.
		math::vec2f m_pivot_vector;	// Vector to pivot the camera about.
		math::quatf m_look_at;		// Current lookat orientation of the camera.
		math::vec3f m_orbit_point;	// Current point that the camera is orbiting about.
		float m_orbit_distance;		// Current distance that the camera is from the orbit point.
		float m_min_distance;		// Minimum distance the camera can be from the orbit point.
		float m_max_distance;		// Maximum distance the camera can be from the rbit point.
		float m_min_orbit_speed;	// Minimum speed that the camera can orbit at.
		float m_max_orbit_speed;	// Maximum speed that the camera can orbit at.
};
}



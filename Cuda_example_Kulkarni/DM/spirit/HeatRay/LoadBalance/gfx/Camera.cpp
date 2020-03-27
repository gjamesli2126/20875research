/*

   Filename : Camera.cpp
   Author   : Cody White and Joe Mahsman
   Version  : 1.0

   Purpose  : Implements a quaternion based camera for use with OpenGL. 

   Change List:

      - 06/12/2009  - Created (Cody White and Joe Mahsman)
	  
	  - 10/14/2009	- Added field-of-view calculations into this class (Cody White)

*/

#include <gfx/Camera.h>
#include <iostream>
using namespace std;

namespace gfx 
{

#define PI 3.141592654

// Default constructor.
Camera::Camera (void)
{
	m_position = math::vec3f (0.0f, 0.0f, 0.0f);
	m_orientation.identity ();
	updateViewMatrix ();
}

// Default destructor.
Camera::~Camera (void)
{
}

// Set the current position of the camera.
void Camera::setPosition (const math::vec3f &position)
{
	m_position = position;
	updateMatrixPosition ();
}

// Get the current position of the camera.
const math::vec3f Camera::getPosition (void) const
{
	return m_position;
}

// Transform the model-view matrix based on the user's input.
void Camera::apply (void) const 
{
#ifdef USE_GL
	// Apply the modelview matrix.
	glMultMatrixf (m_view_matrix.v);
#endif
}

// Move the camera along its local axis.
void Camera::move (float x_distance, float y_distance, float z_distance)
{
	math::quatf conj = m_orientation.conjugate ();
	math::quatf tmp (0.0f, math::vec3f (x_distance, y_distance, z_distance));
	math::quatf result = m_orientation * tmp * conj;

	// Add the resulting axis to the position vector.
	m_position += result.getAxis ();
	updateMatrixPosition ();
}

// Move the camera along its forward vector.
void Camera::move (float negz_distance)
{
	move (0.0f, 0.0f, -negz_distance);
}

// Move the camera along its right vector (x-axis).
void Camera::strafe (float x_distance)
{
	move (x_distance, 0.0f, 0.0f);
}

// Change the pitch of the camera (about the x-axis).
void Camera::pitch (float angle)
{
	math::quatf tmp (angle, math::vec3f (1.0f, 0.0f, 0.0f), true);
	rotateLocal (tmp);
}

// Change the yaw of the camera (about the y-axis).
void Camera::yaw (float angle)
{
	math::quatf tmp (angle, math::vec3f (0.0f, 1.0f, 0.0f), true);
	rotateLocal (tmp);
}

// Change the roll of the camera (about the z-axis).
void Camera::roll (float angle)
{
	math::quatf tmp (angle, math::vec3f (0.0f, 0.0f, 1.0f), true);
	rotateLocal (tmp);
}

// Rotate about the world y axis. value is in radians.
void Camera::turn (float angle)
{
	math::quatf tmp (angle, math::vec3f (0.0f, 1.0f, 0.0f), true);
	rotateLocal (tmp);
}

// Rotate about the world axis.  These are the absolute x, y, and z axis of the world, not local to the camera.
// To perform this operation, the passed in quaternion must be premultiplied into the orientation.
void Camera::rotateWorld (math::quatf q)
{
	m_orientation = q * m_orientation;
	m_orientation.normalize ();
	updateViewMatrix ();
}

// Rotate about the camera's local axis.  To perform this operation, the quaternion passed in must be postmultiplied
// into the orientation.
void Camera::rotateLocal (math::quatf q)
{
	m_orientation = m_orientation * q;
	m_orientation.normalize ();
	updateViewMatrix ();
}

// Update the view matrix.  A local copy of the view matrix is kept so that the quaternion does not need to be
// converted into a matrix each frame (which can be a lengthy process).  This optimization helps if the user 
// has not changed the camera orientation.
void Camera::updateViewMatrix (void)
{
	math::quatf tmp = m_orientation.inverse ();
	tmp.toMatrix (m_view_matrix);
	updateMatrixPosition ();
}

// Render the camera as a coordinate axis.  This is for debugging purposes.
void Camera::debugRender (void) const
{
#ifdef USE_GL
	glDisable(GL_LIGHTING);
	glPushMatrix();

	glMultMatrixf (m_view_matrix.v);
	glScaled (1000, 1000, 1000);

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
#endif
}

// Get the normalized forward vector direction from the current view. This information
// can be obtained from the current model-view matrix.
const math::vec3f Camera::getForwardVector (void) const
{
	return math::vec3f (-m_view_matrix(0, 2), -m_view_matrix(1, 2), -m_view_matrix (2, 2));
}

// Get the normalized up vector direction from the current view.  This information 
// can be obtained from the current model-view matrix.
const math::vec3f Camera::getUpVector (void) const
{
	return math::vec3f (m_view_matrix(0, 1), m_view_matrix(1, 1), m_view_matrix (2, 1));
}

// Get the normalized right vector direction from the current view.  This information 
// can be obtained from the current model-view matrix.
const math::vec3f Camera::getRightVector (void) const
{
	return math::vec3f (m_view_matrix(0, 0), m_view_matrix(1, 0), m_view_matrix (2, 0));
}

// Set the current view perspective of the camera.
void Camera::setPerspective (float fov, float aspect_ratio, float z_near, float z_far)
{
	// Get near, far, right, and left planes.
	GLfloat top = z_near * tanf (fov * PI / 360.0f);
	GLfloat bottom = -top;
	GLfloat left = bottom * aspect_ratio;
	GLfloat right = top * aspect_ratio;  // End of what gluPerspective does

	// Helper variables
	GLfloat right_minus_left = 1.0f / (right - left);
	GLfloat top_minus_bottom = 1.0f / (top - bottom);
	GLfloat far_minus_near 	 = 1.0f / (z_far - z_near);
	GLfloat two_times_near   = 2.0f * z_near;

	// Populate the projection matrix.
	m_projection_matrix(0, 0) = two_times_near * right_minus_left;
	m_projection_matrix(0, 1) = 0.0f;
	m_projection_matrix(0, 2) = 0.0f; 
	m_projection_matrix(0, 3) = 0.0f;

	m_projection_matrix(1, 0) = 0.0f;
	m_projection_matrix(1, 1) = two_times_near * top_minus_bottom;
	m_projection_matrix(1, 2) = 0.0f;
	m_projection_matrix(1, 3) = 0.0f;

	m_projection_matrix(2, 0) = (right + left) * right_minus_left;
	m_projection_matrix(2, 1) = (top + bottom) * top_minus_bottom;
	m_projection_matrix(2, 2) = -((z_far + z_near) * far_minus_near);
	m_projection_matrix(2, 3) = -1.0f;

	m_projection_matrix(3, 0) = 0.0f;
	m_projection_matrix(3, 1) = 0.0f;
	m_projection_matrix(3, 2) = (-two_times_near * z_far) * far_minus_near;
	m_projection_matrix(3, 3) = 0.0f;

#ifdef USE_GL
	// Apply the projection matrix.
	glMatrixMode (GL_PROJECTION);	
	glLoadMatrixf (m_projection_matrix.v);
#endif

	m_aspect_ratio = aspect_ratio;
	m_fov_y = fov;
	m_fov_x = fov * aspect_ratio;
}

// Get the projection matrix.
math::Mat4f Camera::getProjectionMatrix (void) const
{
	return m_projection_matrix;
}

// Get the modelview matrix.
math::Mat4f Camera::getModelViewMatrix (void) const
{
	return m_view_matrix;
}

// Get the aspect ratio of the camera.
float Camera::getAspectRatio (void) const
{
	return m_aspect_ratio;
}

// Get the field of view of the camera (x).
float Camera::getFOVX (void) const
{
	return m_fov_x;
}

// Get the field of view of the camera (y).
float Camera::getFOVY (void) const
{
	return m_fov_y;
}

// Update the position of the camera.
void Camera::updateMatrixPosition (void)
{
	// Set the translation part of the matrix.
	math::vec3f back = getForwardVector () * -1.0f;
	math::vec3f up = getUpVector ();
	math::vec3f right = getRightVector ();

	m_view_matrix (3, 0) = -math::dot (right, m_position);
	m_view_matrix (3, 1) = -math::dot (up, m_position);
	m_view_matrix (3, 2) = -math::dot (back, m_position);
}

}




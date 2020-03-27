#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <hydra/Matrix.h>

namespace gfx
{

class Frustum
{
public:

	enum Visibility
	{
		COMPLETE,
		PARTIAL,
		NONE
	};

	inline void update()
	{
		glGetDoublev(GL_PROJECTION_MATRIX, projection.v);
		glGetDoublev(GL_MODELVIEW_MATRIX, modelview.v);
		modelviewProjection = projection * modelview;
		_planes[0] = modelviewProjection.row(3) - modelviewProjection.row(0); //Right
		_planes[1] = modelviewProjection.row(3) + modelviewProjection.row(0); //Left
		_planes[2] = modelviewProjection.row(3) - modelviewProjection.row(1); //Top
		_planes[3] = modelviewProjection.row(3) + modelviewProjection.row(1); //Bottom
		_planes[4] = modelviewProjection.row(3) - modelviewProjection.row(2); //Far
		_planes[5] = modelviewProjection.row(3) + modelviewProjection.row(2); //Near
		for (int i = 0; i < 6; ++i)
		{
			float length = sqrt(_planes[i].x() * _planes[i].x() +
			                    _planes[i].y() * _planes[i].y() +
								_planes[i].z() * _planes[i].z());
			_planes[i] /= length;
		}

		float inv[16];

		inv[0] = modelview.v[0];
		inv[1] = modelview.v[4];
		inv[2] = modelview.v[8];
		inv[4] = modelview.v[1];
		inv[5] = modelview.v[5];
		inv[6] = modelview.v[9];
		inv[8] = modelview.v[2];
		inv[9] = modelview.v[6];
		inv[10] = modelview.v[10];

		inv[12] = inv[0]*-modelview.v[12]+inv[4]*-modelview.v[13]+inv[8]*-modelview.v[14];
		inv[13] = inv[1]*-modelview.v[12]+inv[5]*-modelview.v[13]+inv[9]*-modelview.v[14];
		inv[14] = inv[2]*-modelview.v[12]+inv[6]*-modelview.v[13]+inv[10]*-modelview.v[14];

		inv[3] = 0.0f;
		inv[7] = 0.0f;
		inv[11] = 0.0f;
		inv[15] = 1.0f;
		_eyePosition = hydra::Vec3(inv[12], inv[13], inv[14]);
	}

	inline Visibility isVisible(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
	{
		hydra::Vec4 points[8];
		points[0] = hydra::Vec4(xmin, ymin, zmin, 1.0f);
		points[1] = hydra::Vec4(xmin, ymin, zmax, 1.0f);
		points[2] = hydra::Vec4(xmin, ymax, zmin, 1.0f);
		points[3] = hydra::Vec4(xmin, ymax, zmax, 1.0f);
		points[4] = hydra::Vec4(xmax, ymin, zmin, 1.0f);
		points[5] = hydra::Vec4(xmax, ymin, zmax, 1.0f);
		points[6] = hydra::Vec4(xmax, ymax, zmin, 1.0f);
		points[7] = hydra::Vec4(xmax, ymax, zmax, 1.0f);
		int planesContaining = 0;
		for (int i = 0; i < 6; ++i)
		{
			int pointsInFront = 0;
			for (int j = 0; j < 8; ++j)
			{
				if (_pointInFront(points[j], _planes[i]))
					++pointsInFront;
			}
			if (pointsInFront == 0)
				return NONE;
			if (pointsInFront == 8)
				++planesContaining;
		}
		if (planesContaining == 6)
			return COMPLETE;
		else
			return PARTIAL;
	}

	inline void computeCorners()
	{
		_farLeftBottom = _intersectPlanes(_planes[4], _planes[1], _planes[3]);
		_farLeftTop = _intersectPlanes(_planes[4], _planes[1], _planes[2]);
		_farRightBottom = _intersectPlanes(_planes[4], _planes[0], _planes[3]);
		_farRightTop = _intersectPlanes(_planes[4], _planes[0], _planes[2]);
	}

	inline hydra::Vec3 eye() { return _eyePosition; }
	inline hydra::Vec3 farLeftBottom() { return _farLeftBottom; }
	inline hydra::Vec3 farLeftTop() { return _farLeftTop; }
	inline hydra::Vec3 farRightBottom() { return _farRightBottom; }
	inline hydra::Vec3 farRightTop() { return _farRightTop; }

	inline float& near() { return _near; }
	inline float& far() { return _far; }

	hydra::Mat4 projection;
	hydra::Mat4 modelview;
	hydra::Mat4 modelviewProjection;
protected:
	hydra::Vec4 _planes[6];
	hydra::Vec3 _eyePosition;
	hydra::Vec3 _farLeftBottom;
	hydra::Vec3 _farLeftTop;
	hydra::Vec3 _farRightBottom;
	hydra::Vec3 _farRightTop;
	float _near;
	float _far;

	inline bool _pointInFront(hydra::Vec4& point, hydra::Vec4& plane)
	{
		return point.dot(plane) > 0;
	}

	inline hydra::Vec3 _intersectPlanes(hydra::Vec4 p1, hydra::Vec4 p2, hydra::Vec4 p3)
	{
		hydra::Vec3 result;
		hydra::Vec3 n1 = hydra::Vec3(p1.x(),p1.y(),p1.z());
		hydra::Vec3 n2 = hydra::Vec3(p2.x(),p2.y(),p2.z());
		hydra::Vec3 n3 = hydra::Vec3(p3.x(),p3.y(),p3.z());
		float d1 = -p1.w();
		float d2 = -p2.w();
		float d3 = -p3.w();
		float denom = n1.dot(n2.cross(n3));
		result = n2.cross(n3) * d1 + n3.cross(n1) * d2 + n1.cross(n2) * d3;
		result /= denom;
		return result;
	}

};

}

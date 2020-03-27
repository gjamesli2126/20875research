// This is a modified version of Hydra's matrix class.
#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <iomanip>

#include <math/Vector.h>

namespace math
{

/**
 * An RxC matrix stored in column-major format.
 */
template <typename T, unsigned int R, unsigned int C>
struct Matrix
{
	// Raw matrix contents in OpenGL format (column major)
	T v[R * C];

	void output() {
		for (int i = 0; i < R*C; i++) {
			printf("%f ", v[i]);
		}
		printf("\n");
	}

	void load(std::ifstream& fin) {
		for (int i = 0; i < R*C; i++) {
			fin >> v[i];
		}
	}

	bool operator==(const Matrix<T,R,C>& right) const
	{
		for (unsigned int i = 0; i < R * C; ++i)
		{
			if (v[i] != right.v[i])
			{
				return false;
			}
		}

		return true;
	}

	/**
	 * Creates a zeroed matrix.
	 */
	Matrix<T, R, C>()
	{
		for (unsigned int i = 0; i < R * C; ++i)
		{
			v[i] = (T)0;
		}
	}

	template <typename L>
	Matrix<T, R, C>(const Matrix<L, R, C>& right)
	{
		for (unsigned int i = 0; i < R * C; ++i)
		{
			v[i] = right.v[i];
		}
	}

	/**
	 * Returns a particular column of the matrix.
	 */
	inline Vector<T, R> col(int c) const
	{
		Vector<T, R> result;
		for (unsigned int i = 0, j = c * R; i < R; ++i, ++j)
		{
			result.v[i] = v[j];
		}
		return result;
	}

	inline T* ptr()
	{
		return v;
	}

	inline const T* c_ptr() const
	{
		return v;
	}

	/**
	 * Returns a particular row of the matrix.
	 */
	inline Vector<T, C> row(int r) const
	{
		Vector<T, C> result;
		for (unsigned int i = 0, j = r; i < C; ++i, j += R)
		{
			result.v[i] = v[j];
		}
		return result;
	}

	/**
	 * Returns the transpose of this matrix.
	 */
	inline Matrix<T, C, R> transpose() const
	{
		Matrix<T, C, R> result;
		for (unsigned int r = 0; r < R; ++r)
		{
			for (unsigned int c = 0; c < C; ++c)
			{
				result(r,c) = (*this)(c,r);
			}
		}
		return result;
	}

	/**
	 * Access a particular matrix element.
	 */
	inline T& operator()(int r, int c)
	{
		return v[c + r * R];
	}

	inline T operator()(int r, int c) const
	{
		return v[c + r * R];
	}

	/**
	 * Post multiplies this matrix by another matrix.
	 */
	template<typename S, unsigned int N>
	inline Matrix<T, R, N> operator*(const Matrix<S, C, N>& right) const
	{
		Matrix<T, R, N> result;
		for (unsigned int r = 0; r < R; ++r)
		{
			for (unsigned int c = 0; c < N; ++c)
			{
				result(r,c) = dot(row(r), right.col(c));
			}
		}
		return result;
	}

	/**
	 * Post multiplies this matrix by a vector.
	 */
	template<typename S>
	inline Vector<S, R> operator*(const Vector<S, C>& right) const
	{
		Vector<S, R> result;
		for (unsigned int r = 0; r < R; ++r)
		{
			result.v[r] = dot(row(r), right);
		}
		return result;
	}

	/**
	 * Post multiplies this matrix by another matrix.
	 */
	template<typename S, unsigned int N>
	inline Matrix<T, R, N> postMult(Matrix<S, C, N>& m)
	{
		Matrix<T, R, N> result;
		for (unsigned int r = 0; r < R; ++r)
		{
			for (unsigned int c = 0; c < N; ++c)
			{
				result(r,c) = dot(row(r), m.col(c));
			}
		}
		return result;
	}

	/**
	 * Pre multiplies this matrix by another matrix.
	 */
	template<unsigned int N>
	inline Matrix<T, N, C> preMult(Matrix<T, N, R>& m)
	{
		Matrix<T, N, C> result;
		for (unsigned int r = 0; r < N; ++r)
		{
			for (unsigned int c = 0; c < C; ++c)
			{
				result(r,c) = dot(m.row(r), col(c));
			}
		}
		return result;
	}

	/**
	 * Copy the contents of the target matrix.
	 */
	template<typename S>
	inline void operator=(const Matrix<S, R, C>& right)
	{
		for (unsigned int i = 0; i < R * C; ++i)
			v[i] = right.v[i];
	}

	static inline Matrix<T, R, C> identity()
	{
		Matrix<T, R, C> result;
		for (int i = 0; i < R && i < C; ++i)
			result(i, i) = 1;
		return result;
	}

	static inline Matrix<T,4,4> translate(T x, T y, T z)
	{
		Matrix<T,4,4> m = identity();
		m(0,3) = x;
		m(1,3) = y;
		m(2,3) = z;
		return m;
	}

	//degrees
	static inline Matrix<T,4,4> rotate(T angle, T x, T y, T z)
	{
		Matrix<T,4,4> m = identity();
		T radAngle = angle / 180.0f * 3.14159f;
		T c = cos(radAngle);
		T s = sin(radAngle);
		T t = 1.0 - c;
		T txx = t * x * x;
		T txy = t * x * y;
		T txz = t * x * z;
		T tyy = t * y * y;
		T tyz = t * y * z;
		T tzz = t * z * z;
		T xs = x * s;
		T ys = y * s;
		T zs = z * s;
		m(0,0) = txx + c;	m(0,1) = txy - zs;	m(0,2) = txz + ys;
		m(1,0) = txy + zs;	m(1,1) = tyy + c;	m(1,2) = tyz - xs;
		m(2,0) = txz - ys;	m(2,1) = tyz + xs;	m(2,2) = tzz + c;
		return m;
	}

	template<class Trait>
	static inline Matrix<T,4,4> rotate(T angle, const Trait& v)
	{
		return rotate(angle, v.x(), v.y(), v.z());
	}

	static inline Matrix<T,4,4> lookAt(T eyeX, 
	                                       T eyeY, 
										   T eyeZ,
	                                       T lookAtX, 
										   T lookAtY, 
										   T lookAtZ,
										   T upX,
										   T upY,
										   T upZ)
	{
	}

	template<class Trait>
	static inline Matrix<T,4,4> lookAt(const Trait& eye,
	                                       const Trait& lookAt,
										   const Trait& up)
	{
		Matrix<T,4,4> m = Matrix<T,4,4>::identity();
		Trait f = normalize(lookAt - eye);
		Trait u = normalize(up);
		Trait r = cross(f, u);
		u = cross(r, f);
		m(0,0) = r.x(); m(0,1) = r.y(); m(0,2) = r.z();
		m(1,0) = u.x(); m(1,1) = u.y(); m(1,2) = u.z();
		m(2,0) = -f.x(); m(2,1) = -f.y(); m(2,2) = -f.z();
		Matrix<T,4,4> translation = translate(-eye.x(), -eye.y(), -eye.z());
		return m * translation;
	}
};

/**
 * Returns an inverted matrix determined via Gaussian Elimination.
 */
template <typename T, unsigned int R, unsigned int C>
inline Matrix<T, R, C> invert(const Matrix<T, R, C>& matrix)
{
	Matrix<T, R, 2 * C> workspace;
	for (int row = 0; row < R; ++row)
	{
		for (int col = 0; col < C; ++col)
		{
			workspace(row, col) = matrix.v[row + col * R];
		}
	}
	for (int rc = 0; rc < R; ++rc)
	{
		workspace(rc, rc + C) = (T)1.0;
	}
	int row = 0;
	int col = 0;
	while (row < R && col < C)
	{
		int maxRow = row;
		for (int i = row + 1; i < R; ++i)
		{
			if (abs(workspace(i, col)) > abs(workspace(maxRow, col)))
				maxRow = i;
		}
		if (workspace(maxRow, col) != 0)
		{
			for (int c = 0; c < 2 * C; ++c)
			{
				std::swap(workspace(row, c), workspace(maxRow, c));
			}
			T divisor = workspace(row, col);
			for (int c = 0; c < 2 * C; ++c)
			{
				workspace(row, c) /= divisor;
			}
			for (int i = 0; i < R; ++i)
			{
				if (i == row)
					continue;
				T multiplier = workspace(i, col);
				for (int c = 0; c < 2 * C; ++c)
				{
					workspace(i, c) -= workspace(row, c) * multiplier;
				}
			}
		}
		++row;
		++col;
	}
	Matrix<T,R,C> result;
	for (int row = 0; row < R; ++row)
	{
		for (int col = 0; col < C; ++col)
		{
			result(row, col) = workspace(row, col + C);
		}
	}
	return result;
}

// Implementing this as a free function inside of this header removes the
// circular dependency between Matrix and Vector.
template<typename T, unsigned int N, unsigned int C>
inline Vector<T, C> operator*(const Vector<T,N>& left, const Matrix<T, N, C>& right)
{
	Vector<T, C> result;
	for (int i = 0; i < N; ++i)
	{
		result.v[i] = dot(left, right.col(i));
	}
	return result;
}

template<typename T, unsigned int R, unsigned int C>
inline std::ostream& operator<<(std::ostream& out, const Matrix<T,R,C>& m)
{
	out << "{" << std::endl << " ";
	for (int i = 0; i < R; ++i)
	{
		for (int j = 0; j < C; ++j)
			#if 1
				out << std::setw(20) << m(i, j);
			#else
				// uncomment for printing matrix setting code
				out << "mv[" << j * 4 + i << "] = " << std::setw(10) << m(i, j) << "; "; //temp for results analysis
			#endif
		out << std::endl << " ";
	}
	out << "}";
	return out;
}

typedef Matrix<float,4,4> Mat4f;
typedef Matrix<double,4,4> Mat4d;

} // namespace math

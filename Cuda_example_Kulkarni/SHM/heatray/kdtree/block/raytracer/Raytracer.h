/*

   Filename : Raytracer.h
   Author   : Cody White
   Version  : 1.0

   Purpose  : Class which performs ray tracing on a mesh. 

   Change List:

      - 12/23/2009  - Created (Cody White)

 */

#pragma once

#include "gfx/gfx.h"

#include <shared/Ray.h>
#include <shared/IntersectInfo.h>
#include <shared/Light.h>
#include <shared/SearchResult.h>
#include <util/Timer.h>
#include <math/Vector.h>
#include <math/Quaternion.h>
#include <kdtree/TriangleKDTree.h>
#include <bvh/PhotonBVH.h>
#include <string.h>
#include <util/random.h>
#include <math/Constants.h>

class Block;
class BlockStack;
class IntermediaryState;
class IntermediaryBlock;

typedef math::Vector <float, 3> rvec;


/**
 * Ray tracer class.
 */
class Raytracer
{
public:
	/**
	 * Default constructor.
	 */
	Raytracer (void)
	{
		m_tree = NULL;
		m_screen_width = 0;
		m_screen_height = 0;
		m_colors = NULL;
		m_num_photons = 0;
		m_num_photons_divisor = (float)1.0;
		m_photons_to_gather = 0;
		m_max_gather_distance = (float)0.0;
	}

	/**
	 * Destructor.
	 */
	~Raytracer (void)
	{
		m_tree = NULL;
		if (m_colors != NULL)
		{
			delete [] m_colors;
		}
	}

	void initialize (TriangleKDTree *tree, size_t photons_per_light, size_t num_to_gather, float max_gather_distance, size_t caustic_density);
	void start (int screen_width, int screen_height, rvec forward, rvec up, rvec right, rvec origin, std::vector <Light > *lights);
	void generatePhotonMap (void);
	void raytrace (void);
	void render (void) const;
	void CastPhotonsBlock(int start, int end); 

private:

	/**
	 * Payload struct, this struct is passed along with the ray
	 * so that functions using ray properties know about where
	 * the ray has been.
	 */
	struct Payload
	{
		rvec multiplier;		// Amount of enery left in the ray in each color spectrum.
		float index_of_refraction;	// Index of refraction for the current material the ray is inside.
		rvec color;				// Accumulated color of this ray.
		int num_bounces;		// Number of bounces this ray has made so far.
		bool inside;			// Flag to determine if the ray is inside of an object or not.
	};

	void photonCreate();
	void photonMap();
	void photonMap_block();

	void castPhoton (Photon& p, IntersectInfo &info, Ray &ray, std::vector <Photon >* new_points);
	void castPhoton_prologue(const Photon& photon, IntersectInfo &info, Ray &ray, Block *_block);
	void castPhoton_main(BlockStack *_stack);
	void castPhoton_mainAutotune(BlockStack *_stack, _Autotuner *_autotuner);
	void castPhoton_epilogue(const Photon& p, IntersectInfo &info, Ray &ray, std::vector <Photon >* new_points);

	struct RayPoint {
		RayPoint(Ray r, Payload p, int i) {
			ray = r;
			payload = p;
			index = i;
		}

		Ray ray;
		Payload payload;
		int index;
	};

	struct DiffusePoint {
		DiffusePoint(rvec d, rvec h, rvec n, int i) {
			diffuse_light = d;
			hit_point = h;
			normal = n;
			index = i;
		}
		rvec diffuse_light;
		rvec hit_point;			// Point on the triangle intersected by the ray.
		rvec normal;			// Normal coordinate at the hit point.
		int index;
	};


	void doTrace ();
	void doTrace_block();

	float castRay (Ray &ray, Payload &p, int index, std::vector <RayPoint >* new_points, std::vector<DiffusePoint>* diffuse_points);
	float castRay_epilogue(Ray &ray, Payload &p, IntersectInfo &info, int index, std::vector <RayPoint >* new_points,
			std::vector<DiffusePoint>* diffuse_points, IntermediaryState *_inter_state);
	void castRay_main(BlockStack *_stack);
	void castRay_prologue(Ray &ray, Payload &p, IntersectInfo &info, IntermediaryState *_inter_state, Block *_block);

	float gammaCorrect (float color);
	rvec diffuse  (rvec d, rvec hit_point, rvec normal);
	void specular (rvec s, IntersectInfo &info, Ray &ray, Payload &p, int index, std::vector <RayPoint >* new_points);
	rvec diffuseDirection (rvec &normal, rvec &forward, RandomFloat *random);

	// Member variables.
	TriangleKDTree			*m_tree;					// KD-Tree which contains the triangle data for the scene.
	PhotonBVH				m_indirect_photon_map;		// BVH of stored indirect photons for computing global illumination.
	PhotonBVH				m_caustic_photon_map;		// BVH of stored caustic photons for computing caustics in the scene.
	GLfloat 					*m_colors;					// Color buffer to use for population by the rays and rendering by OpenGL.
	int 						m_screen_width;				// Width of the screen in pixels.
	int 						m_screen_height;			// Screen height in pixels.
	float 							m_half_screen_width;		// Half of the screen width in pixels.
	float 							m_half_screen_height;		// Half of the screen height in pixels.
	size_t 						m_num_threads;				// Number of threads to run.
	Thread 						*m_threads;					// Group of threads to perform tracing.  There are as many threads as cores on the machine.
	rvec 						m_w, m_u, m_v;				// Basis vectors of the image plane. w = up, u = left, v = up.
	rvec 						m_origin;					// Eye point of the user.
	std::vector <Light >		*m_lights;					// List of all of the lights in the scene.
	util::Timer					m_timer;					// Used for benchmarking.
	size_t						m_num_photons;				// Number of photons that each light emits.
	float	 						m_num_photons_divisor;		// 1 / m_num_photons;
	std::vector <Photon >	m_start_photon_list;		// The list of photons which originate from the lights.
	std::vector <Photon >	m_indirect_photon_list;		// The list of indirect photons which have bounced around the scene.
	size_t						m_photons_to_gather;		// The number of photons to gather during the gather pass.
	float							m_max_gather_distance;		// Maximum distance to gather photons.
	size_t						m_caustic_density;			// Density value for caustic photons.

	vector<RandomFloat*> m_randoms;

	// Debug
	std::vector <Photon *> m_tmp_photons;
	math::vec3f m_tmp_hit;

	size_t m_num_rays;
	size_t m_num_cast_ray_calls;
	size_t m_total_photons;
	size_t m_num_cast_photon_calls;
	uint64_t m_bvh_nodes_traversed;
};



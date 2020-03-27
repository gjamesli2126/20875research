/*
 * Raytracer.cpp
 *
 *  Created on: Oct 30, 2012
 *      Author: yjo
 */
#include<time.h>
#include "raytracer/Raytracer.h"
//#define ONE_PHASE
/**
 * Initialize this class for use.
 */
void Raytracer::initialize (GAL* tree, size_t photons_per_light, size_t num_to_gather, float max_gather_distance, size_t caustic_density)
{
	m_tree = tree;
	m_photons_to_gather = num_to_gather;
	m_max_gather_distance = max_gather_distance;
	m_num_photons = photons_per_light;
	m_num_photons_divisor = (float)1.0 / m_num_photons;
	m_caustic_density = caustic_density;
}

/**
 * Start the ray tracing and photon mapping process.
 * First, if needed, a photon map will be created.
 * Second, the scene will be ray traced using the photon map.
 * @param screen_width Width of the screen in pixels.
 * @param screen_height Height of the screen in pixels.
 * @param forward Forward vector of the camera.
 * @param up Up vector of the camera.
 * @param right Right vector of the camera.
 * @param origin Eye-position of the viewer.
 * @param lights List of lights to cast photons from.
 */
void Raytracer::start (int screen_width, int screen_height, rvec forward, rvec up, rvec right, rvec origin, std::vector <Light > *lights) {
	m_screen_width  = screen_width;
	m_screen_height = screen_height;
	m_half_screen_width  = (float)0.5 * m_screen_width;
	m_half_screen_height = (float)0.5 * m_screen_height;
	m_lights = lights;

	// Setup the world axis.
	m_w = forward * (float)-1.0;
	m_u = right;
	m_v = up;
	m_origin = origin;

	m_num_rays = 0;
	m_num_cast_ray_calls = 0;
	m_total_photons = 0;
	m_num_cast_photon_calls = 0;
	m_bvh_nodes_traversed = 0;
	m_kdtree_nodes_traversed =0;

	// The first phase of the rendering process is to generate the photon map.
#if defined(USE_GL) || defined(FULL_RENDER)
	//std::cout << "Casting photons..." << std::endl;
	generatePhotonMap ();
	//std::cout << "DONE!" << std::endl;

	// The second phase of the process is to ray trace the scene using the photon map.
	//std::cout << "Starting ray tracing..." << std::endl;
	raytrace ();
#else
#ifndef RAY_TRACE
	//std::cout << "Casting photons..." << std::endl;
	generatePhotonMap ();
	//std::cout << "DONE!" << std::endl;
#else
	// The second phase of the process is to ray trace the scene using the photon map.
	//std::cout << "Starting ray tracing..." << std::endl;
	raytrace ();
#endif
#endif

}

/**
 * Generate the photon map.
 */
void Raytracer::generatePhotonMap (void) {
	// Check to see if we already have a photon map of the scene.
	// If so, we don't need to regenerate one since the lighting
	// and geometry of the scene is static.
	// 	DISABLED TO REPEAT TESTS
	//if (m_indirect_photon_map.exists ()) return;

	// First allocate the correct number of photons for the scene.
	// This can be calculated as the number of photons each light
	// emits * the number of lights.
	size_t num_photons = m_lights->size () * m_num_photons;
	m_start_photon_list.resize (num_photons);
	m_randoms.resize(num_photons);
	photonCreate();
	photonMap();

	m_start_photon_list.clear ();
	//cout << "m_indirect_photon_list: " << m_indirect_photon_list.size() << endl;
	m_indirect_photon_map.build (m_indirect_photon_list);
	m_indirect_photon_list.clear ();
	for (int i = 0; i < m_randoms.size(); i++) {
		delete m_randoms[i];
	}
}

/**
 * Raytrace the scene.
 */
void Raytracer::raytrace (void) {
	if (m_colors != NULL) {
		delete [] m_colors;
	}

	// Allocate the colors array (R, G, B).
	m_colors = new GLfloat [m_screen_width * m_screen_height * 3];

	// Clear the colors array.
	memset (m_colors, 0.0f, m_screen_width * m_screen_height * 3 * sizeof (GLfloat));

	// Setup the barrier.
	//pthread_barrier_init (&m_thread_barrier, NULL, m_num_threads);

	// Setup the ticket counter.
	m_timer.start ();
	doTrace();
	double render_time = m_timer.stop ();
	std::cout << "Render time = " << render_time / 60000000.0 << " minutes." << std::endl;
	cout << "m_num_rays: " << m_num_rays << endl;
	cout << "m_num_cast_ray_calls: " << m_num_cast_ray_calls << endl;
}

/**
 * Render the current raytraced scene.
 */
void Raytracer::render (void) const {
#ifdef USE_GL
	glDisable (GL_LIGHTING);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	gluOrtho2D (0, 1, 0, 1);
	glRasterPos2d (0, 0);
	glDrawPixels (m_screen_width, m_screen_height, GL_RGB, GL_FLOAT, m_colors);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix ();

	glEnable (GL_LIGHTING);
#endif
}


/**
 * Create a group of photons assigned to this thread.
 */
void Raytracer::photonCreate() {
	// Create the photons.
	int end = m_start_photon_list.size();
	for (int i = 0; i < end; ++i) {
		int light = i * m_num_photons_divisor;

		// Set the photon properties.
		m_start_photon_list[i].position = (*m_lights)[light].position;
		m_start_photon_list[i].power = (*m_lights)[light].power * m_num_photons_divisor;

		m_randoms[i] = new RandomFloat(i);
		// Generate 2 random angles for a direction.
		float angle_x = m_randoms[i]->interval(-(*m_lights)[light].fov_x, (*m_lights)[light].fov_x);
		float angle_y = m_randoms[i]->interval(-(*m_lights)[light].fov_y, (*m_lights)[light].fov_y);

		// Create quaternions out of these 2 new angles with the
		// right and up vectors, respectively.
		::math::Quaternion<float> q_x, q_y;
		q_x.createFromAxisAngle (angle_x, (*m_lights)[light].right);
		q_y.createFromAxisAngle (angle_y, (*m_lights)[light].up);

		::math::Quaternion<float> q = q_x * q_y;

		// Rotate the forward vector of the light by this new quaternion
		// and that will be the direction of the photon.
		m_start_photon_list[i].direction = ::math::normalize (q.rotate ((*m_lights)[light].forward));

		m_start_photon_list[i].random = m_randoms[i];
		m_start_photon_list[i].seedForRandom = i;
	}

	//if (!Harness::get_sort_flag()) 
	{
		int num_lights = m_lights->size();
		std::vector <Photon > start_photon_list;
		start_photon_list.resize(num_lights * m_num_photons);
		for (int i = 0; i < m_num_photons; i++) {
			for (int j = 0; j < num_lights; j++) {
				start_photon_list[i * num_lights + j] = m_start_photon_list[j * m_num_photons + i];
			}
		}
		for (int i = 0; i < end; i++) {
			m_start_photon_list[i] = start_photon_list[i];
		}
	}
}

/**
 * Create the list of bounced photons for storing in the photon BVH using threads.
 */
void Raytracer::photonMap() {
	std::vector<Photon> globalPhotons;
	std::vector <Photon > new_photons;
	std::vector <Photon >* points;
	std::vector <Photon >* new_points;
	m_total_photons = m_start_photon_list.size();
	points = &m_start_photon_list;
	new_points = &new_photons;

	double starttime, endtime, consumedtime=0;
#ifdef STATISTICS2
	long int numStagesExecuted=0, totalStagesExecuted=0;
	long int numRaysTraced=0;
#endif

	while (!points->empty()) 
	{
#ifdef STATISTICS2
		numRaysTraced += points->size();
#endif
		uint64_t cnt = 0;
		int end = points->size();
		int endIndex = end;
		int startIndex = 0;
		int pid = m_tree->GetOpt_ProcRank();

#ifdef LOAD_BALANCE
		int numProcs = m_tree->GetOpt_NumProcs();
		int numPoints = end;
		//every process gets [startIndex, endIndex) chunk of rayVector.
		//Block distribution of points among processes
		startIndex = (pid) * (numPoints/numProcs);
		endIndex = (pid == (numProcs-1))?(numPoints):((pid+1) * (numPoints/numProcs));
		if(numPoints < numProcs)
		{
			if(pid >= numPoints)
			{
				startIndex = 0;
				endIndex = 0;
			}
			else
			{
				startIndex = pid;
				endIndex = pid+1;
			}
		}
#endif

		std::vector<Ray> rayVector(end);
		std::vector<IntersectInfo> infoVector(end);
		for (int i = 0; i < end; i++) {
			Photon& p = (*points)[i];
			IntersectInfo info;
			// Create a ray out of the photon so that it can be
			// cast through the TriangleKDTree.
			Ray ray(p.position,p.direction);
			rayVector[i] =ray;
			infoVector[i] =info;
		}
		/*rayVector.erase(rayVector.begin()+1,rayVector.end());
		infoVector.erase(infoVector.begin()+1,infoVector.end());*/
		m_tree_visitor = new TriangleKDTreeVisitor(pid,m_tree,rayVector, infoVector);
#ifdef LOAD_BALANCE
		m_tree_visitor->GALVisitor_AssignWorkItems(startIndex, endIndex);
#endif
		m_tree->GAL_Synchronize();
		starttime = clock();

		castPhotons(m_start_photon_list, rayVector, infoVector, new_photons, startIndex, endIndex);
#ifdef TRACK_TRAVERSALS
#ifndef LOAD_BALANCE
		if(m_tree->GetOpt_ProcRank() == 0)
#endif
		{
			for(int i=startIndex;i<endIndex;i++)
			//for(int i=startIndex;i<1;i++)
			{
				//printf("ray %d: %d\n",i,rayVector[i].num_nodes_traversed);
#ifdef STATISTICS2
				numStagesExecuted +=rayVector[i].numStagesExecuted;
#endif
				cnt+=rayVector[i].num_nodes_traversed;
			}
			/*if(m_tree->GetOpt_ProcRank() == 0)
				printf("%d \t %llu \t %6.1f\n", end, cnt, (float)cnt / end);*/
			m_kdtree_nodes_traversed += cnt;
		}
#endif
		endtime = clock();
		consumedtime += (endtime-starttime);
		delete m_tree_visitor;
		m_tree_visitor = NULL;
		m_start_photon_list.clear();
#ifdef ONE_PHASE
		break;
#endif
#ifndef LOAD_BALANCE
		m_start_photon_list = new_photons;
#else
		m_tree->AllGather(new_photons, globalPhotons);
		m_start_photon_list = globalPhotons;
		globalPhotons.clear();
#endif
		new_photons.clear();
		rayVector.clear();
		infoVector.clear();
	}
	m_tree->GAL_PrintGraph();
	double total_consumedtime=0.0;
	uint64_t total_nodes_traversed=0;
	m_tree->GAL_AggregateResults(m_kdtree_nodes_traversed, consumedtime, total_nodes_traversed, total_consumedtime);
#ifdef STATISTICS2
	m_tree->GAL_AggregateResults_Generic(numStagesExecuted, totalStagesExecuted);
	if(m_tree->GetOpt_ProcRank() == 0)
		printf("Average stages executed per point: %f\n",totalStagesExecuted/(float)numRaysTraced);
#endif
	//cout << "m_total_photons: " << m_total_photons << endl;
	//cout << "m_num_cast_photon_calls: " << m_num_cast_photon_calls << endl;
}

void Raytracer::castPhotons(std::vector<Photon>& m_cast_photon_list, std::vector<Ray>& rayVector, std::vector<IntersectInfo>& infoVector, std::vector<Photon>& new_photons, long int startIndex, long int endIndex)
{
	m_tree_visitor->Traverse();
	while(true)
	{
#ifndef LOAD_BALANCE
		if(m_tree->GetOpt_ProcRank() == 0)
#endif
		{
#ifdef PERFCOUNTERS
			m_tree->GAL_StartComputeTimer();
#endif
			for(int i=0;i<rayVector.size();i++)
			{
				if(i == endIndex)
					break;
				if(i >= startIndex)
				{
					if(rayVector[i].intersects) 
					{
						// This photon has intersected the geometry.
						rvec diffuse 		= (infoVector[i].material).diffuse;
						rvec specular 		= (infoVector[i].material).specular;
						rvec transmissive 	= (infoVector[i].material).transmissive;

						if ((infoVector[i].material).texture.hasData ()) {
							::math::Vector <float, 4> tex = (infoVector[i].material).texture.getPixelColor (infoVector[i].tex_coord[0], infoVector[i].tex_coord[1]);
							diffuse *= tex.xyz ();
						}

						float pd = (diffuse[0] + diffuse[1] + diffuse[2]) / 3;
						float ps = (specular[0] + specular[1] + specular[2]) / 3;
						float pt = (transmissive[0] + transmissive[1] + transmissive[2]) / 3;

						// Perform Russian-Roulette for this photon.
						//float value = m_cast_photon_list[i].random->get();
						Photon photon = m_cast_photon_list[i];
						float value;
						int numInv = (m_randoms[m_cast_photon_list[i].seedForRandom])->getNumInvocations();
						int reqInv = photon.numInvocations;
						do
						{
						value = (m_randoms[m_cast_photon_list[i].seedForRandom])->get();
						numInv++;
						}while(numInv <= reqInv);

						if (value <= pd) 
						{
							// Diffuse reflection.
							photon.position = infoVector[i].hit_point + (infoVector[i].normal * (float)0.00001);
							photon.power *= diffuse * pd;
							photon.normal = infoVector[i].normal;

							if (photon.caustic == true) {
								for (size_t i = 0; i < m_caustic_density; ++i) {
									m_indirect_photon_list.push_back (photon);
								}
							} else {
								m_indirect_photon_list.push_back (photon);
							}
							
							rvec tmp = ::math::normalize ((infoVector[i].triangle).m_vertices[0] - infoVector[i].hit_point);
							photon.direction = diffuseDirection (infoVector[i].normal, tmp, (m_randoms[m_cast_photon_list[i].seedForRandom]));
							photon.caustic = false;
							photon.numInvocations = (m_randoms[m_cast_photon_list[i].seedForRandom])->getNumInvocations();
							new_photons.push_back(photon);
#ifndef LOAD_BALANCE
							//GAL_BroadCastObject(photon,m_tree);
							//m_tree->GAL_Test(photon);
#endif
						} 
						else if (value <= pd + ps) 
						{
							// Specular reflection.
							photon.position = infoVector[i].hit_point + (infoVector[i].normal * (float)0.00001);
							photon.normal = infoVector[i].normal;
							photon.power *= specular * ps;
							photon.direction = ::math::normalize (::math::reflect (photon.direction, infoVector[i].normal));
							photon.caustic = true;
							photon.numInvocations = (m_randoms[m_cast_photon_list[i].seedForRandom])->getNumInvocations();
							new_photons.push_back(photon);
#ifndef LOAD_BALANCE
							//GAL_BroadCastObject(photon,m_tree);
#endif
						}
						else if (value <= pd + ps + pt) 
						{
							// Transmission

							// Use Fresnel to determine what to do with this photon.
							float new_ior = (float)0.0;
							rvec refract_normal = rvec::zero ();

							if (photon.inside == false)
							{
								// Entering object.
								new_ior = (infoVector[i].material).index_of_refraction;
								refract_normal = infoVector[i].normal;
							} 
							else if (photon.inside == true) 
							{
								// Leaving object.
								new_ior = (float)1.0; // Air
								refract_normal = infoVector[i].normal * (float)-1.0; // Reverse the normal because we are inside of the object.
							}

							photon.position = infoVector[i].hit_point + (refract_normal * (float)0.00001);

							rvec new_dir;
							float kr = (float)0.0;
							float kt = (float)0.0;
							::math::fresnelRefract (photon.direction, refract_normal, photon.index_of_refraction, new_ior, new_dir, kr, kt);

							photon.caustic = true;

							// Generate a new random to use with the fresnel kt.
							//value = m_cast_photon_list[i].random->get();
							value = (m_randoms[m_cast_photon_list[i].seedForRandom])->get();
							photon.numInvocations = (m_randoms[m_cast_photon_list[i].seedForRandom])->getNumInvocations();
							if (value <= kr) 
							{
								// Reflect this ray instead of transmit it.
								photon.power *= (specular * kr) * ps;
								photon.normal = infoVector[i].normal;
								photon.direction = ::math::normalize (::math::reflect (photon.direction, infoVector[i].normal));
								photon.inside = false;
								photon.index_of_refraction = (float)1.0;
								new_photons.push_back(photon);
							} 
							else 
							{
								// Refract this photon.
								photon.power *= (transmissive * kt) * pt;
								photon.direction = new_dir;
								photon.normal = refract_normal;
								photon.inside = !photon.inside;
								photon.index_of_refraction = new_ior;
								new_photons.push_back(photon);
							}
#ifndef LOAD_BALANCE
							//GAL_BroadCastObject(photon,m_tree);
#endif
						} 
						else 
						{
							// Absorbtion.
						}
					}
				}
			}
#ifndef LOAD_BALANCE
			//GAL_BroadCastMessage(m_tree,MESSAGE_DONEKDTREE);
#endif

#ifdef PERFCOUNTERS
			m_tree->GAL_StopComputeTimer();
#endif
			break;
#ifndef LOAD_BALANCE
			//SendTerminateToAllProcs 
#endif
		}
#ifndef LOAD_BALANCE
		else
		{
			/*Photon p;
			bool dataReceived = GAL_ReceiveObject(p,m_tree);
			if(!dataReceived)
			{
				break;
			}
			new_photons.push_back(p);*/
			break;
		}
#endif
	}
}


/**
 * Threaded ray tracing function.
 */
void Raytracer::doTrace () {
	float d = m_half_screen_width / tanf ((::math::PI / (float)180.0) * (float)0.5 * (float)45.0);

	uint64_t cnt = 0;
	std::vector <RayPoint > rays, new_rays;
	std::vector <RayPoint >* points;
	std::vector <RayPoint >* new_points;

	rvec *results = new rvec[m_screen_height * m_screen_width];
	memset (results, 0.0f, m_screen_width * m_screen_height * sizeof (rvec));
	for (int row = 0; row < m_screen_height; ++row) {
		for (int c = 0; c < m_screen_width; ++c) { // For each column in this row.
			int index = m_screen_width * row + c;
			Ray ray;
			ray.origin = m_origin;
			float x = c - m_half_screen_width;
			float y = row - m_half_screen_height;
			ray.direction = ::math::normalize (m_u * x + m_v * y - m_w * d);
			Payload p;
			p.multiplier = rvec ((float)1.0, (float)1.0, (float)1.0);
			p.index_of_refraction = (float)1.0;
			p.color = rvec ((float)0.0, (float)0.0, (float)0.0);
			p.num_bounces = 0;
			p.inside = false;

			rays.push_back(RayPoint(ray, p, index));
			m_num_rays++;
		}
	}

	points = &rays;
	new_points = &new_rays;
	std::vector <DiffusePoint > diffuse_points;
	uint64_t sum = 0;
	while (!points->empty()) {
		uint64_t cnt = 0;
		int end = points->size();
		for (int i = 0; i < end; i++) {
			castRay ((*points)[i].ray, (*points)[i].payload, (*points)[i].index, new_points, &diffuse_points);
			results[(*points)[i].index] += (*points)[i].payload.color;
#ifdef TRACK_TRAVERSALS
			cnt += (*points)[i].ray.num_nodes_traversed;
#endif
		}
#ifdef TRACK_TRAVERSALS
		sum += cnt;
		printf("%d \t %llu \t %6.1f\n", end, cnt, (float)cnt / end);
#endif
		std::vector <RayPoint >* tmp = points;
		points = new_points;
		new_points = tmp;
		new_points->clear();
	}
#ifdef TRACK_TRAVERSALS
	cout << "total nodes traversed: " << sum << endl;
#endif

#if defined(USE_GL) || defined(FULL_RENDER)
	//for (const std::vector<DiffusePoint>::iterator it = diffuse_points.begin(); it < diffuse_points.end(); it++) {
	for (int i = 0; i < diffuse_points.size(); i++) {
		DiffusePoint *it = &diffuse_points[i];
		results[it->index] += diffuse(it->diffuse_light, it->hit_point, it->normal);
	}
#ifdef TRACK_TRAVERSALS
	cout << "bvh nodes traversed: " << m_bvh_nodes_traversed << endl;
#endif

	for (int row = 0; row < m_screen_height; ++row) {
		for (int c = 0; c < m_screen_width; ++c) { // For each column in this row.
			int i = m_screen_width * row + c;
			int index = i * 3;
			rvec color;
			color[0] = m_colors[index];
			color[1] = m_colors[index + 1];
			color[2] = m_colors[index + 2];

			color = results[i];

			m_colors[index]     = color.x();
			m_colors[index + 1] = color.y();
			m_colors[index + 2] = color.z();
		}
	}
#endif
	delete [] results;
}

/**
 * Cast a ray.
 * @param ray Ray to cast.
 * @param p Payload associated with this ray.
 */
float Raytracer::castRay (Ray &ray, Payload &p, int index, std::vector <RayPoint >* new_points, std::vector<DiffusePoint>* diffuse_points) {
printf("CastRay returning empty handed\n");
#if 0
	m_num_cast_ray_calls++;

	// Terminate after 10 bounces.
	if (p.num_bounces > 10) {
		return (float)0.0;
	}

	float time = (float)0;
	IntersectInfo info;

	// determine if this ray intersects the triangle mesh.
	m_tree_visitor->intersect (ray, info);
	if (ray.intersects) {
		time = info.time;
		if (!(info.time < (float)HUGE_VAL)) {
			// do nothing
		} else {
			// Diffuse
			rvec current_diffuse = p.multiplier * info.material->diffuse;
			if (math::dot (current_diffuse, current_diffuse) > (float)0.0001) {
				if (info.material->texture.hasData ()) {
					math::Vector <float, 4> tex = info.material->texture.getPixelColor (info.tex_coord[0], info.tex_coord[1]);
					current_diffuse *= tex.xyz ();
				}
				diffuse_points->push_back(DiffusePoint(current_diffuse, info.hit_point, info.normal, index));
				//diffuse (current_diffuse, info, ray, p);
			}

			// Specular
			rvec current_specular = p.multiplier * info.material->specular;
			if (math::dot (current_specular, current_specular) > (float)0.0001) {
				specular (current_specular, info, ray, p, index, new_points);
			}

			// Transmissive.
			rvec current_transmissive = p.multiplier * info.material->transmissive;
			if (math::dot (current_transmissive, current_transmissive) > (float)0.0001) {
				// Do any scenes have this? Ignore and bail out for now.
				cout << "transmissive!" << endl;
				exit(0);
				//p.color += refract (current_transmissive, info, ray, p);
			}
		}
	} else {
		// If the ray missed, set the color to cornflower blue.
		//p.color += rvec ((float)0.390625f, (float)0.5703125f, (float)0.92578125) * p.multiplier;
	}

	return time;
#endif
}


/**
 * Gamma correct a certain color.
 * @param color Color to correct.
 */
float Raytracer::gammaCorrect (float color)
{
	if (color <= 0.0031308f)
	{
		return color * 12.92f;
	}

	return 1.055f * pow (color, 0.4166667f) - 0.055f;
}

/**
 * Add a diffuse color to the ray.
 * @param d Amount of light for diffuse calculations.
 * @param info Intersection info until this point.
 * @param ray Incoming ray.
 * @param p Ray payload.
 */
rvec Raytracer::diffuse  (rvec d, rvec hit_point, rvec normal) {
	rvec radiance = rvec::zero ();
	std::vector <SearchResult > photons;
#ifdef TRACK_TRAVERSALS
	m_bvh_nodes_traversed += m_indirect_photon_map.knn (hit_point, m_photons_to_gather, m_max_gather_distance, photons);
#else
	m_indirect_photon_map.knn (hit_point, m_photons_to_gather, m_max_gather_distance, photons);
#endif
	if (photons.size ()) {
		float r = photons[photons.size () - 1].distance;
		const float kernel_filter = (float)1.1;
		for (size_t i = 0; i < photons.size (); ++i) {
			float dot_photon = ::math::dot (-photons[i].photon->direction, normal);
			if (dot_photon > (float)0.0) {
				float filter_weight = (float)1.0 - photons[i].distance / (kernel_filter * r);
				radiance += photons[i].photon->power * dot_photon * filter_weight;
			}
		}

		radiance *= d;
		radiance /= (M_PI * r * r) * ((float)1.0 - ((float)2.0 / ((float)3.0 * kernel_filter)));
	}
	return radiance;
}

/**
 * Cast a specular ray.
 * @param s Amount of light being reflected.
 * @param info Intersection infomration until this point.
 * @param ray Incoming ray.
 * @param p Ray payload.
 */
void Raytracer::specular (rvec s, IntersectInfo &info, Ray &ray, Payload &p, int index, std::vector <RayPoint >* new_points) {
	// Create a specular ray and fire it off.
	Payload pay;
	pay.multiplier = s;
	pay.color = rvec::zero ();
	pay.index_of_refraction = p.index_of_refraction;
	pay.num_bounces = p.num_bounces + 1;
	pay.inside = p.inside;

	Ray spec;
	spec.origin = info.hit_point + (info.normal * (float)0.00001);
	spec.direction = ::math::normalize (::math::reflect (ray.direction, info.normal));

	// Perform specular highlighting.
	for (size_t i = 0; i < m_lights->size (); ++i) {
		rvec light_dir = ::math::normalize ((*m_lights)[i].position - info.hit_point);
		rvec r = ::math::normalize (::math::reflect (light_dir, info.normal));
		float n_dot_l = ::math::dot (info.normal, light_dir);
		float r_dot_view = ::math::dot (r, m_w * (float)-1.0);
		if (r_dot_view > (float)0.0 && n_dot_l > (float)0.0) {
			p.color += (rvec::one () * std::max ((float)0.0, powf  (r_dot_view, (info.material).specular_exponent)))* n_dot_l;
		}
	}

	//castRay (spec, pay);
	new_points->push_back(RayPoint(spec, pay, index));
}

/**
 * Cast a transmissive and (possibly) a specular ray.
 * @param r Amount of light being reflected/refracted.
 * @param info Intersection infomration until this point.
 * @param ray Incoming ray.
 * @param p Ray payload.
 */
//	math::Vector <float, 3> refract  (rvec r, IntersectInfo &info, Ray &ray, Payload &p) {
//		float new_ior = (float)0.0;
//		rvec refract_normal = rvec::zero ();
//
//		if (p.inside == false) {
//			// Entering object.
//			new_ior = info.material->index_of_refraction;
//			refract_normal = info.normal;
//		} else {
//			// Leaving object.
//			new_ior = (float)1.0; // Air
//			refract_normal = info.normal * (float)-1.0; // Reverse the normal because we are inside of the object.
//		}
//
//		rvec new_dir;
//		float kr = (float)0.0;
//		float kt = (float)0.0;
//		math::fresnelRefract (math::normalize (ray.direction), refract_normal, p.index_of_refraction, new_ior, new_dir, kr, kt);
//		rvec color = rvec::zero ();
//		if (kr > (float)0.0) {
//			color += specular (r * kr, info, ray, p);
//		}
//
//		if (kt > (float)0.0) {
//			Ray r_ray;
//			r_ray.origin = info.hit_point + (refract_normal * (float)0.0001);
//			r_ray.direction = new_dir;
//			Payload pay;
//			pay.multiplier = r * kt;
//			pay.color = rvec::zero ();
//			pay.index_of_refraction = new_ior;
//			pay.num_bounces = p.num_bounces + 1;
//			pay.inside = !p.inside;
//			float time = castRay (r_ray, pay);
//
//			// Apply Beer's Law.
//			rvec absorbance = info.material->diffuse * (float)0.15 * -time;
//			rvec transparency = rvec (expf (absorbance.r ()), expf (absorbance.b ()), expf (absorbance.g ()));
//
//			color += pay.color * transparency;
//		}
//
//		return color;
//	}

/**
 * Calculate a random diffuse direction.
 * @param normal Normal to the hit point.
 * @param forward Forward vector to perform the rotation on.
 */
rvec Raytracer::diffuseDirection (rvec &normal, rvec &forward, RandomFloat *random) {
	rvec dir;
	float randx = random->interval(-::math::PI, ::math::PI);
	float randy = random->interval(-::math::PI_OVER_TWO, ::math::PI_OVER_TWO);
	::math::Quaternion<float> q_x, q_y;
	q_x.createFromAxisAngle (randx, forward);
	q_y.createFromAxisAngle (randy, normal);

	::math::Quaternion<float> q = q_x * q_y;
	dir = ::math::normalize (q.rotate (normal));
	return dir;
}



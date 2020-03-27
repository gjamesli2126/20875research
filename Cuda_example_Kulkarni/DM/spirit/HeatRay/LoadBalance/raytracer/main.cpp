/*

   Filename : main.cpp
   Author   : Cody White
   Version  : 1.0

   Purpose  : Main driver for a ray tracer with an OpenGL previewer. 

   Change List:

      - 12/20/2009  - Created (Cody White)
 */

#include "gfx/gfx.h"

#include <util/Timer.h>
#include <raytracer/World.h>
#include "raytracer/Params.h"
#include <boost/mpi.hpp>
#include<boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include<boost/graph/distributed/adjacency_list.hpp>
#ifdef PAPI
#include"papi.h"
void handle_error(int retval)
{
	printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	exit(1);
}
#endif



// window title
char* title = (char*)"Heat Ray";

// window size
//GLint screen_width = 800;
//GLint screen_height = 800;
//GLint half_screen_width = 400;
//GLint half_screen_height = 400;
GLint screen_width = 400;
GLint screen_height = 400;
GLint half_screen_width = screen_width / 2;
GLint half_screen_height = screen_height / 2;


// window position
const GLint screen_pos_x = 50;
const GLint screen_pos_y = 50;

#ifdef USE_GL
// Main class to handle all screen drawing and actions
World world;
#endif

// Timer used for updates
util::Timer timer;

// Input keys array
bool keys [255] = { false };
bool special_keys [255] = { false };

const GLfloat light_ambient[]  = { 0.0f, 0.0f, 0.0f, 0.0f };
const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };



#ifdef USE_GL
GLvoid display();
GLvoid initGL (int argc, char **argv);
GLvoid normalKeyPressed(unsigned char key, int x, int y);
GLvoid normalKeyReleased(unsigned char key, int x, int y);
GLvoid specialKeyPressed(int key, int x, int y);
GLvoid specialKeyReleased(int key, int x, int y);
GLvoid resize(GLsizei width, GLsizei height);
GLvoid update();
GLvoid mouseMove (GLint x, GLint y);

// Call down to the the Screen class to take care of the drawing.
GLvoid display (void)
{
	world.render ();
}

// Initialize OpenGL
GLvoid initGL (int argc, char **argv)
{
	resize(screen_width, screen_height);

	glClearColor(0.0, 0.0, 0.0, 1.0);

	glEnable(GL_DEPTH_TEST);
	glEnable (GL_LIGHTING);
	glEnable (GL_LIGHT0);
	glEnable (GL_NORMALIZE);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

	world.init (argc, argv);
}

// Keyboard callback for standard keys.
GLvoid normalKeyPressed(unsigned char key, int x, int y)
{
	world.normal_keys [key] = true;
}

GLvoid normalKeyReleased(unsigned char key, int x, int y)
{
	world.normal_keys [key] = false;
}

GLvoid specialKeyPressed(int key, int x, int y)
{
	world.special_keys [key] = true;
}

GLvoid specialKeyReleased(int key, int x, int y)
{
	world.special_keys [key] = false;
}

GLvoid resize(GLsizei width, GLsizei height)
{
	if (0 == height)
	{
		height = 1;
	}

	glViewport(0, 0, width, height);

	// SET WORLD PROJECTION 45 degrees
	world.setProjection (width, height, 45.0f, 0.1f, 10000.0f);

	screen_width = width;
	screen_height = height;
	half_screen_width = width * 0.5f;
	half_screen_height = height * 0.5f;
}

// Idle state.  Post a redraw to the event loop.
GLvoid update()
{
	static GLboolean started = false;
	GLdouble dt;

	// start the timer if it has not been started already
	if (!started)
	{
		timer.start();
		started = true;
	}

	// get the change in time
	dt = timer.stop() / 1000000.0;
	timer.start();

	world.update (dt);

	glutPostRedisplay();
}

GLvoid mouseMove (GLint x, GLint y)
{
	if (x != half_screen_width || y != half_screen_height)
	{
		world.rotateCamera (-(x - half_screen_width), -(y - half_screen_height));
		glutWarpPointer (half_screen_width, half_screen_height);
	}
}

// Main function
int main (int argc, char **argv)
{
	checkParams(argc);

	srand(0);

	screen_width = atoi(argv[ScreenWidth]);
	screen_height = atoi(argv[ScreenHeight]);

			// initialize glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE |
			GLUT_DEPTH | GLUT_MULTISAMPLE |
			GLUT_STENCIL);

	// setup and create window
	glutInitWindowSize(screen_width, screen_height);
	glutInitWindowPosition(screen_pos_x, screen_pos_y);
	int window = glutCreateWindow(title);

	// initialize everything
	glewInit();
	initGL (argc, argv);

	// setup callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(normalKeyPressed);
	glutKeyboardUpFunc(normalKeyReleased);
	glutSpecialFunc(specialKeyPressed);
	glutSpecialUpFunc(specialKeyReleased);
	glutReshapeFunc(resize);
	glutIdleFunc(update);
	//glutPassiveMotionFunc (mouseMove);

	//glutSetCursor (GLUT_CURSOR_NONE);
	//glutIgnoreKeyRepeat (1);

	glutMainLoop();

	return 0;
}

#else

int main (int argc, char **argv) {

#ifdef PAPI
	int retval, EventSet=PAPI_NULL;
	long long values[4] = {(long long)0, (long long)0, (long long)0, (long long)0};

	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if(retval != PAPI_VER_CURRENT)
		handle_error(retval);
	
	retval = PAPI_multiplex_init();
	if (retval != PAPI_OK) 
		handle_error(retval);
	retval = PAPI_create_eventset(&EventSet);
	if (retval != PAPI_OK) 
		handle_error(retval);
 
	// Add Total L2Cache Misses 
	retval = PAPI_add_event(EventSet, PAPI_L2_TCM);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// Total L1 cache accesses = total memory accesses. Needed for computing L2 miss rate. On Qstruct, there are 2 layers of cache. 
	retval = PAPI_add_event(EventSet, PAPI_L2_TCA);
	if (retval != PAPI_OK) 
		handle_error(retval);

	retval = PAPI_set_multiplex(EventSet);
	if (retval != PAPI_OK) 
		handle_error(retval);

	// TOTAL cycles 
	retval = PAPI_add_event(EventSet, PAPI_TOT_CYC);
	if (retval != PAPI_OK) 
		handle_error(retval);
	
	// TOTAL instructions 
	retval = PAPI_add_event(EventSet, PAPI_TOT_INS);
	if (retval != PAPI_OK) 
		handle_error(retval);

#endif

	boost::mpi::environment env(argc, argv);
	mpi_process_group pg;
	mpi::communicator comm;
	int numProcs = num_processes(pg);
	int pid = process_id(pg);

	checkParams(argc);

	srand(0);

	World world(pg);
	world.init (argc, argv);

	/*double startTime, endTime;
	startTime = clock();*/
#ifdef PAPI
	retval = PAPI_start(EventSet);
	if (retval != PAPI_OK) handle_error(retval);
#endif
	world.start();
#ifdef PAPI
	/* Stop the counters */
	retval = PAPI_stop(EventSet, values);
	if (retval != PAPI_OK) 
		handle_error(retval);

	float avgMissRate, missRate = values[0]/(double)(values[1]);
	float avgCPI, CPI = values[2]/(double)(values[3]);
	reduce(comm,missRate, avgMissRate, std::plus<float>(),0);
	reduce(comm,CPI, avgCPI, std::plus<float>(),0);
	if(pid==0)
		printf("Average L2 Miss Rate:%f Average CPI:%f\n",avgMissRate/numProcs,avgCPI/numProcs);
	//printf("%d: L2 Cache Miss Rate:%f CPI:%f\n",pid, values[0]/(double)(values[1]),values[2]/(double)(values[3]));

#endif
	/*endTime=clock();
	double consumedTime = endTime - startTime;
	if(pid == 0)
	{
		world.printstats();
		std::cout<<"time consumed: "<<consumedTime/CLOCKS_PER_SEC<<" seconds"<<std::endl;
	}*/

	return 0;
}
#endif

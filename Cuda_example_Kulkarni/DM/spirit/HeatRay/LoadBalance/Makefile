CPP=mpic++
all: PM

aggr: CFLAGS+=-DMESSAGE_AGGREGATION
aggr: all

INSTALLED_HOME = /home/min/a/hegden/Research/HeatRay/PreReqs
GLEW_DIR = $(INSTALLED_HOME)/GLEW
FREEIMAGE_DIR = $(INSTALLED_HOME)/FreeImage
ASSIMP_DIR = $(INSTALLED_HOME)/assimp--2.0.863-sdk
BOOST_DIR = /home/min/a/hegden/IMP/boost_1_55_0
#BOOST_DIR = /home/min/a/hegden/ECE573/Installed/boost_1_54_0
BOOST_LIBS = $(BOOST_DIR)/stage/lib
PAPI_DIR=/home/min/a/hegden/Research/PAPI
PAPI_DEP_DIR=/opt/intel/current/lib/intel64

CFLAGS += -I$(GLEW_DIR)/include -I$(FREEIMAGE_DIR)/Source -I$(ASSIMP_DIR)/include
CFLAGS += -I$(BOOST_DIR) -I$(PAPI_DIR)/include
#PROFILING_FLAGS = -DSPAD_2 -DSTATISTICS #-DPAPI #-DSTATISTICS2
CFLAGS += -DTRACK_TRAVERSALS -DNDEBUG $(PROFILING_FLAGS)
LIBS += -lfreeimage -L$(FREEIMAGE_DIR)/lib -lassimp -L$(ASSIMP_DIR)/lib  -lboost_mpi -lboost_graph_parallel -lboost_system -L$(BOOST_LIBS) -L$(PAPI_DIR)/lib -lpapi -L$(PAPI_DEP_DIR)

BIN_DEPS = Camera.o Framebuffer.o Program.o Shader.o Skybox.o Texture.o Uniform.o ThreadManager.o \
	file.o main.o GAL.o Raytracer.o World.o TriangleKDTreeVisitor.o PhotonBVH.o Triangle.o Ray.o TriangleKDTreeNode.o clops.o timer.o

ifeq ($(USE_GL),1)
	CFLAGS += -DUSE_GL
	LIBS += -lglut -lGL -lGLU -lGLEW  -L$(GLEW_DIR)/lib64 -L/usr/X11R6/lib
	# this was an attempt to get OpenGL working on Yeti03, eventually got HeatRay to work without OpenGL at all 
	#LIBS += -L../../../Installed/Mesa-8.0.4/lib64
	
endif
	 

PM: $(BIN_DEPS) 
	#$(CPP) $(LIBS) $(BOOST_LIBS)/libboost_mpi.so $(BOOST_LIBS)/libboost_graph_parallel.so $(BOOST_LIBS)/libboost_system.so  -o heatray $(BIN_DEPS)  
	$(CPP) $(LIBS) -o PM $(BIN_DEPS)  
Camera.o: gfx/Camera.h gfx/Camera.cpp
	$(CPP) $(CFLAGS) -I. -c gfx/Camera.cpp
Framebuffer.o: gfx/Framebuffer.h gfx/Framebuffer.cpp
	$(CPP) $(CFLAGS) -I. -c gfx/Framebuffer.cpp
Program.o: 
	$(CPP) $(CFLAGS) -I. -c gfx/Program.cpp
Shader.o: 
	$(CPP) $(CFLAGS) -I. -c gfx/Shader.cpp
Skybox.o: 
	$(CPP) $(CFLAGS) -I. -c gfx/Skybox.cpp
Texture.o: 
	$(CPP) $(CFLAGS) -I. -c gfx/Texture.cpp
Uniform.o: 
	$(CPP) $(CFLAGS) -I. -c gfx/Uniform.cpp
ThreadManager.o: 
	$(CPP) $(CFLAGS) -I. -c thread/ThreadManager.cpp
file.o: 
	$(CPP) $(CFLAGS) -I. -c util/file.cpp
GAL.o: kdtree/GAL.cpp kdtree/GAL.h
	$(CPP) $(CFLAGS)  -I. -c kdtree/GAL.cpp
Raytracer.o: raytracer/Raytracer.cpp raytracer/Raytracer.h
	$(CPP) $(CFLAGS) -I. -c raytracer/Raytracer.cpp
World.o:raytracer/World.cpp  raytracer/World.h
	$(CPP) $(CFLAGS) -I. -c raytracer/World.cpp
PhotonBVH.o:
	$(CPP) $(CFLAGS) -I. -c bvh/PhotonBVH.cpp
Triangle.o: 
	$(CPP) $(CFLAGS) -I. -c gfx/Triangle.cpp
Ray.o:
	$(CPP) $(CFLAGS) -I. -c shared/Ray.cpp
TriangleKDTreeNode.o:
	$(CPP) $(CFLAGS) -I. -c kdtree/TriangleKDTreeNode.cpp
TriangleKDTreeVisitor.o:kdtree/TriangleKDTreeVisitor.cpp kdtree/TriangleKDTreeVisitor.h
	$(CPP) $(CFLAGS) -I. -c kdtree/TriangleKDTreeVisitor.cpp
clops.o: util/clops.cpp util/clops.h
	$(CPP) $(CFLAGS) -I. -c util/clops.cpp
timer.o: util/timer.cpp util/timer.h
	$(CPP) $(CFLAGS) -I. -c util/timer.cpp
	
main.o: raytracer/main.cpp raytracer/World.h raytracer/Raytracer.h shared/Ray.h shared/Light.h kdtree/TriangleKDTreeVisitor.h \
	gfx/VertexBuffer.h util/random.h shared/Photon.h
	$(CPP) $(CFLAGS) -I. -c raytracer/main.cpp
	
clean:
	rm -f *~ PM *.o *.csv
	rm -rf .obj

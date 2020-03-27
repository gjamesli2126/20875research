all: heatray

include ../../../Makefile.common


INSTALLED_HOME = /home/min/a/hegden/Research/HeatRay/PreReqs
GLEW_DIR = $(INSTALLED_HOME)/GLEW
FREEIMAGE_DIR = $(INSTALLED_HOME)/FreeImage
ASSIMP_DIR = $(INSTALLED_HOME)/assimp--2.0.863-sdk
BOOST_DIR=/home/min/a/hegden/Installed/boost_1_55_0/

CFLAGS += -I$(GLEW_DIR)/include -I$(FREEIMAGE_DIR)/Source -I$(ASSIMP_DIR)/include
CFLAGS += -I$(BOOST_DIR)
LIBS += -lfreeimage -L$(FREEIMAGE_DIR) -lassimp -L$(ASSIMP_DIR)/lib 

BIN_DEPS = Camera.o Framebuffer.o Program.o Shader.o Skybox.o Texture.o Uniform.o ThreadManager.o \
	file.o main.o Raytracer.o World.o TriangleKDTree.o PhotonBVH.o Triangle.o
BIN_DEPS += $(HARNESS_OBJ)
BIN_DEPS += app_blocks.o blocks.o interstate.o autotuner.o

ifeq ($(USE_GL),1)
	CFLAGS += -DUSE_GL
	LIBS += -lglut -lGL -lGLU -lGLEW  -L$(GLEW_DIR)/lib -L/usr/X11R6/lib
	# this was an attempt to get OpenGL working on Yeti03, eventually got HeatRay to work without OpenGL at all 
	#LIBS += -L../../../Installed/Mesa-8.0.4/lib64
	
endif
	 

heatray: $(BIN_DEPS) 
	$(CC) -o heatray $(BIN_DEPS)  $(LIBS)


Camera.o: gfx/Camera.h gfx/Camera.cpp
	$(CC) $(CFLAGS) -I. -c gfx/Camera.cpp
Framebuffer.o: gfx/Framebuffer.cpp
	$(CC) $(CFLAGS) -I. -c gfx/Framebuffer.cpp
Program.o: 
	$(CC) $(CFLAGS) -I. -c gfx/Program.cpp
Shader.o: 
	$(CC) $(CFLAGS) -I. -c gfx/Shader.cpp
Skybox.o: 
	$(CC) $(CFLAGS) -I. -c gfx/Skybox.cpp
Texture.o: 
	$(CC) $(CFLAGS) -I. -c gfx/Texture.cpp
Uniform.o: 
	$(CC) $(CFLAGS) -I. -c gfx/Uniform.cpp
ThreadManager.o: 
	$(CC) $(CFLAGS) -I. -c thread/ThreadManager.cpp
file.o: 
	$(CC) $(CFLAGS) -I. -c util/file.cpp
	
Raytracer.o: raytracer/Raytracer.cpp raytracer/Raytracer.h
	$(CC) $(CFLAGS) -I. -c raytracer/Raytracer.cpp
World.o: raytracer/World.cpp raytracer/World.h
	$(CC) $(CFLAGS) -I. -c raytracer/World.cpp
TriangleKDTree.o: kdtree/TriangleKDTree.cpp kdtree/TriangleKDTree.h
	$(CC) $(CFLAGS) -I. -c kdtree/TriangleKDTree.cpp
PhotonBVH.o: bvh/PhotonBVH.cpp bvh/PhotonBVH.h
	$(CC) $(CFLAGS) -I. -c bvh/PhotonBVH.cpp
Triangle.o: gfx/Triangle.cpp gfx/Triangle.h
	$(CC) $(CFLAGS) -I. -c gfx/Triangle.cpp
	
blocks.o: $(BLOCK_DEPS)
	$(CC) $(CFLAGS) -I. -o blocks.o -c $(BLOCK_SRC)
interstate.o: $(INTERSTATE_DEPS)
	$(CC) $(CFLAGS) -I. -o interstate.o -c $(INTERSTATE_SRC)
app_blocks.o: app_blocks.cpp app_blocks.h
	$(CC) $(CFLAGS) -I. -c app_blocks.cpp

main.o: raytracer/main.cpp raytracer/World.h raytracer/Raytracer.h shared/Ray.h shared/Light.h kdtree/TriangleKDTree.h \
	gfx/VertexBuffer.h util/random.h shared/Photon.h
	$(CC) $(CFLAGS) -I. -c raytracer/main.cpp

autotuner.o: autotuner.h autotuner.cpp
	$(CC) $(CFLAGS) -I. -c autotuner.cpp

clean:
	rm -f *~ heatray *.o *.csv
	rm -rf .obj

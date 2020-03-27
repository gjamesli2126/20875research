#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
CPP=mpic++#-std=c++11
PROFILING_FLAGS = -DTRACK_TRAVERSALS#-DPAPI
PAPI_HOME=/home/min/a/hegden/Research/PAPI
FLAGS=-lpthread -lm 
FLAGS+=$(PROFILING_FLAGS)

FLAGS=-DNDEBUG $(PROFILING_FLAGS)
INCLUDES=-I/home/min/a/hegden/IMP/boost_1_55_0/ 
INCLUDELIB=/home/min/a/hegden/IMP/boost_1_55_0/stage/lib

ifeq ($(PAPI),1)
FLAGS += -DPAPI
FLAGS +=$(PAPI_HOME)/lib/libpapi.so
INCLUDES += -I$(PAPI_HOME)/include
endif

ifeq ($(TRAVERSAL_PROFILE),1)
FLAGS += -DTRAVERSAL_PROFILE
endif

all:knearestneighbor

knearestneighbor: nn.c kdtree.c
	$(CPP) $(FLAGS) $(INCLUDES) $(INCLUDELIB)/libboost_mpi.so $(INCLUDELIB)/libboost_graph_parallel.so $(INCLUDELIB)/libboost_system.so -o $@ $^


clean:
	rm -rf *.o knearestneighbor *~

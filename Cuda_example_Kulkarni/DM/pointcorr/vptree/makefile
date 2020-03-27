#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
CPP=mpic++
#uncomment below line to turn profiling on.
#PROFILING_FLAGS = -DSTATISTICS
ifeq ($(PAPI),1)
PROFILING_FLAGS += -DPAPI
endif
ifeq ($(TRAVERSAL_PROFILE),1)
PROFILING_FLAGS += -DTRAVERSAL_PROFILE
endif
LINK_FLAGS=$(PAPI_HOME)/lib/libpapi.so
FLAGS= -DNDEBUG $(PROFILING_FLAGS)
PAPI_HOME=/home/min/a/hegden/Research/PAPI
INCLUDES=-I/home/min/a/hegden/IMP/boost_1_55_0/ -I$(PAPI_HOME)/include/ 
INCLUDELIB=/home/min/a/hegden/IMP/boost_1_55_0/stage/lib

all : PC

aggr: FLAGS+=-DMESSAGE_AGGREGATION
aggr: all

PC:CorrelationVisitor.cpp TestGL.cpp GAL.cpp timer.cpp
	$(CPP) $(FLAGS) $(INCLUDES) $(INCLUDELIB)/libboost_mpi.so $(INCLUDELIB)/libboost_graph_parallel.so $(INCLUDELIB)/libboost_system.so $(LINK_FLAGS) -o $@ $^
clean:
	rm PC
	


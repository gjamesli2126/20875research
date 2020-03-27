#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
CPP=mpic++
#uncomment below two lines to turn profiling on.
PROFILING_FLAGS = -DSTATISTICS
ifeq ($(PAPI),1)
PROFILING_FLAGS += -DPAPI
endif
ifeq ($(TRAVERSAL_PROFILE),1)
PROFILING_FLAGS += -DTRAVERSAL_PROFILE
endif
LINK_FLAGS=$(PAPI_HOME)/lib/libpapi.so
#FLAGS=-DNDEBUG -DDYNAMIC_LB $(PROFILING_FLAGS)
#FLAGS=-DNDEBUG -DHYBRID_BUILD $(PROFILING_FLAGS)
FLAGS=-DNDEBUG $(PROFILING_FLAGS)
PAPI_HOME=/home/min/a/hegden/Research/PAPI
INCLUDES=-I/home/min/a/hegden/IMP/boost_1_55_0/ -I$(PAPI_HOME)/include/ 
INCLUDELIB=/home/min/a/hegden/IMP/boost_1_55_0/stage/lib

all : BH

aggr: FLAGS+=-DMESSAGE_AGGREGATION
aggr: all

BH:CorrelationVisitor.cpp TestGL.cpp GAL.cpp timer.cpp AuxFn.cpp
	$(CPP) $(FLAGS) $(INCLUDES) $(INCLUDELIB)/libboost_mpi.so $(INCLUDELIB)/libboost_graph_parallel.so $(INCLUDELIB)/libboost_system.so $(LINK_FLAGS) -o $@ $^

clean:
	rm BH
	


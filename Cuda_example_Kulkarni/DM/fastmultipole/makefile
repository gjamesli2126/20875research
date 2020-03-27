#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
CPP=mpic++

PROFILING_FLAGS = -DSTATISTICS #-DSPAD_2 #-DTRAVERSAL_PROFILE  
PAPI_HOME=/home/min/a/hegden/Research/PAPI
FLAGS=-DNDEBUG $(PROFILING_FLAGS)
INCLUDES=-I/home/min/a/hegden/IMP/boost_1_55_0/ -I$(PAPI_HOME)/include/ 
INCLUDELIB=/home/min/a/hegden/IMP/boost_1_55_0/stage/lib

ifeq ($(PAPI),1)
FLAGS += -DPAPI
FLAGS +=$(PAPI_HOME)/lib/libpapi.so
endif
ifeq ($(TRAVERSAL_PROFILE),1)
FLAGS += -DTRAVERSAL_PROFILE
endif

all : FMM

aggr: FLAGS+=-DMESSAGE_AGGREGATION
aggr: all

FMM:CorrelationVisitor.cpp TestGL.cpp GAL.cpp timer.cpp AuxFn.cpp
	$(CPP) $(FLAGS) $(INCLUDES) $(INCLUDELIB)/libboost_mpi.so $(INCLUDELIB)/libboost_graph_parallel.so $(INCLUDELIB)/libboost_system.so -o $@ $^

clean:
	rm FMM
	


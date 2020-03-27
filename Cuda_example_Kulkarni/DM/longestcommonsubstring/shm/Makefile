#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
CPP=mpic++
INCLUDES=-I/home/min/a/hegden/IMP/boost_1_55_0/ -I$(PAPI_HOME)/include/ -I.
INCLUDELIB=/home/min/a/hegden/IMP/boost_1_55_0/stage/lib

CFLAGS =-std=c++0x $(INCLUDES)
LDFLAGS = $(INCLUDELIB)/libboost_mpi.so $(INCLUDELIB)/libboost_graph_parallel.so $(INCLUDELIB)/libboost_system.so

all: lcs

lcs: main.cpp 
	$(CPP) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ 

clean:
	-rm -f *~ lcs *.o *.csv


#*************************************************************************************************
# Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
BENCHMARK = barneshut

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS = $(BENCHMARK).o $(HARNESS_OBJ) bh_types.o 

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

$(BENCHMARK).o: $(BENCHMARK).cpp
	$(CC) $(CFLAGS) -I. -c $(BENCHMARK).cpp

bh_types.o: bh_types.cpp bh_types.h
	$(CC) $(CFLAGS) -I. -c bh_types.cpp
		
clean:
	rm -f *~ $(BENCHMARK) *.o *.csv

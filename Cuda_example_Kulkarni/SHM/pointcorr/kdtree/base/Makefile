#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
BENCHMARK = pointcorr

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS = $(BENCHMARK).o $(HARNESS_OBJ) pc_types.o

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

$(BENCHMARK).o: $(BENCHMARK).cpp
	$(CC) $(CFLAGS) -I. -c $(BENCHMARK).cpp

pc_types.o: pc_types.cpp pc_types.h
	$(CC) $(CFLAGS) -I. -c pc_types.cpp

clean:
	rm -f *~ $(BENCHMARK) *.o *.csv

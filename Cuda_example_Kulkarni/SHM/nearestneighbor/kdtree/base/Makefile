#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
BENCHMARK = knn_boundingbox

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS = $(BENCHMARK).o $(HARNESS_OBJ) nn_types.o

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

$(BENCHMARK).o: $(BENCHMARK).cpp
	$(CC) $(CFLAGS) -I. -c $(BENCHMARK).cpp

nn_types.o: nn_types.cpp nn_types.h
	$(CC) $(CFLAGS) -I. -c nn_types.cpp

clean:
	rm -f *~ $(BENCHMARK) *.o *.csv

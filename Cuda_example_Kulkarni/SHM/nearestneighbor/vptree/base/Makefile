#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
BENCHMARK = knn_vptree

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS = $(BENCHMARK).o $(HARNESS_OBJ)

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

$(BENCHMARK).o: $(BENCHMARK).cpp
	$(CC) $(CFLAGS) -I. -c $(BENCHMARK).cpp

clean:
	rm -f *~ $(BENCHMARK) *.o *.csv

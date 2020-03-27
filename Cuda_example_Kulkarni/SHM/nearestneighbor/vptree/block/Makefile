#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
BENCHMARK = knn

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS = $(BENCHMARK)_vptree.o $(HARNESS_OBJ)

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

$(BENCHMARK)_vptree.o: $(BENCHMARK)_vptree.cpp block.h interstate.h
	$(CC) $(CFLAGS) -I. -c $(BENCHMARK)_vptree.cpp
	
clean:
	rm -f *~ $(BENCHMARK) *.o *.csv

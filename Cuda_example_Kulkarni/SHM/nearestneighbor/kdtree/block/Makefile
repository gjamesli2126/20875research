#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
BENCHMARK = knn

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS = $(BENCHMARK)_boundingbox.o $(HARNESS_OBJ) nn_types.o  blocks.o app_blocks.o

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

$(BENCHMARK)_boundingbox.o: $(BENCHMARK)_boundingbox.cpp
	$(CC) $(CFLAGS) -I. -c $(BENCHMARK)_boundingbox.cpp

nn_types.o: nn_types.cpp nn_types.h
	$(CC) $(CFLAGS) -I. -c nn_types.cpp
	
blocks.o: $(BLOCK_DEPS)
	$(CC) $(CFLAGS) -I. -o blocks.o -c $(BLOCK_SRC)

app_blocks.o: app_blocks.cpp app_blocks.h
	$(CC) $(CFLAGS) -I. -c app_blocks.cpp
	
clean:
	rm -f *~ $(BENCHMARK) *.o *.csv

#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
BENCHMARK = knearestneighbor

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS =   nn.o kdtree.o

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

nn.o: nn.c nn.h util_common.h
	$(CC) $(CFLAGS) -c nn.c

kdtree.o: kdtree.c nn.h util_common.h
	$(CC) $(CFLAGS) -c kdtree.c

clean:
	rm -rf *.o $(BENCHMARK) *~

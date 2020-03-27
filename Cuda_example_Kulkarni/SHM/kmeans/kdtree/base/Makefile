#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************
BENCHMARK = kmeans

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS =   kmeans_kernel.o functions.o main.o

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) $(CFLAGS) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

kmeans_kernel.o: kmeans_kernel.cpp kmeans_kernel.hpp
	$(CC) $(CFLAGS) -c kmeans_kernel.cpp

functions.o: functions.cpp functions.hpp
	$(CC) $(CFLAGS) -c functions.cpp

main.o: main.cpp common.h functions.hpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm -rf *.o $(BENCHMARK) *~

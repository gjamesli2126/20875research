#*************************************************************************************************
#* Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
#* Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
BENCHMARK = knn

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS = BBT_kernel.o functions.o main.o $(HARNESS_OBJ) app_blocks.o blocks.o

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

main.o: main.cpp
	$(CC) $(CFLAGS) -I. -c main.cpp

BBT_kernel.o: BBT_kernel.cpp BBT_kernel.hpp
	$(CC) $(CFLAGS) -I. -c BBT_kernel.cpp

functions.o: functions.cpp functions.hpp
	$(CC) $(CFLAGS) -I. -c functions.cpp

app_blocks.o: app_blocks.cpp app_blocks.h
	$(CC) $(CFLAGS) -I. -c app_blocks.cpp

blocks.o: $(BLOCK_DEPS)
	$(CC) $(CFLAGS) -I. -o blocks.o -c $(BLOCK_SRC)


clean:
	rm -rf *.o $(BENCHMARK) *~

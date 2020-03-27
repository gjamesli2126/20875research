#*************************************************************************************************
#* Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
#* Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
BENCHMARK = knearestneighbor

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS =   BBT_kernel.o functions.o main.o

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

BBT_kernel.o: BBT_kernel.cpp BBT_kernel.hpp
	$(CC) $(CFLAGS) -c BBT_kernel.cpp

functions.o: functions.cpp functions.hpp
	$(CC) $(CFLAGS) -c functions.cpp

main.o: main.cpp common.h functions.hpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm -rf *.o $(BENCHMARK) *~

#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
CC=g++

all:balltree

balltree: BBT_kernel.o functions.o main.o
	$(CC) -lm -lpthread -o balltree BBT_kernel.o functions.o main.o

BBT_kernel.o: BBT_kernel.cpp BBT_kernel.hpp
	$(CC) -c BBT_kernel.cpp

functions.o: functions.cpp functions.hpp
	$(CC) -c functions.cpp

main.o: main.cpp common.h functions.hpp
	$(CC) -c main.cpp

clean:
	rm -rf *.o balltree *~

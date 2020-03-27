#*************************************************************************************************
# Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
BENCHMARK = lcs

all: $(BENCHMARK)
include ../../Makefile.common

CFLAGS +=-std=c++0x
LDFLAGS +=-lrt 

$(BENCHMARK): $(BENCHMARK).cpp 
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f *~ $(BENCHMARK) *.o *.csv


#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
BENCHMARK = fmm

all: $(BENCHMARK)

include ../Makefile.common

$(BENCHMARK): $(BENCHMARK).cpp
	$(CC) $(CFLAGS) $(LIBS) -I. -o $@ $^

clean:
	rm -f *~ $(BENCHMARK) *.o *.csv

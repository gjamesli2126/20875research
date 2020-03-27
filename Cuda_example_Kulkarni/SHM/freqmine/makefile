#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
BENCHMARK = fpgrowth

ifeq ($(TRAVERSAL_PROFILE),1)
CFLAGS += -DTRAVERSAL_PROFILE
endif

all: $(BENCHMARK)
include ../Makefile.common

$(BENCHMARK): main.cpp $(BENCHMARK).cpp
	$(CC) $(CFLAGS) $(LIBS) -o $@ $^

clean:
	rm -f *~ $(BENCHMARK) *.o *.csv
	


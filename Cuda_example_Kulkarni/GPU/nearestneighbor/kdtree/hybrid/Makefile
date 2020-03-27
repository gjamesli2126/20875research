#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
include ../../../common/cuda.mk

BENCHMARK = nn

all: $(BENCHMARK)

$(BENCHMARK): 	$(BENCHMARK).o \
				$(BENCHMARK)_functions.o \
				$(BENCHMARK)_mem.o \
				$(BENCHMARK)_pre_kernel.o \
				$(BENCHMARK)_kernel.o
	$(NVCC) -o $(BENCHMARK) $(BENCHMARK).o \
							$(BENCHMARK)_functions.o \
							$(BENCHMARK)_mem.o \
							$(BENCHMARK)_pre_kernel.o \
							$(BENCHMARK)_kernel.o

$(BENCHMARK).o: $(BENCHMARK).cu
	$(NVCC) $(NVCC_OPTIONS) -c $(BENCHMARK).cu

$(BENCHMARK)_functions.o: $(BENCHMARK)_functions.cu $(BENCHMARK)_functions.h
	$(NVCC) $(NVCC_OPTIONS) -c $(BENCHMARK)_functions.cu

$(BENCHMARK)_mem.o: $(BENCHMARK)_mem.cu $(BENCHMARK)_mem.h
	$(NVCC) $(NVCC_OPTIONS) -c $(BENCHMARK)_mem.cu

$(BENCHMARK)_pre_kernel.o: $(BENCHMARK)_pre_kernel.cu $(BENCHMARK)_pre_kernel.h
	$(NVCC) $(NVCC_OPTIONS) -c $(BENCHMARK)_pre_kernel.cu
	
$(BENCHMARK)_kernel.o: $(BENCHMARK)_kernel.cu $(BENCHMARK)_kernel.h
	$(NVCC) $(NVCC_OPTIONS) -c $(BENCHMARK)_kernel.cu


clean:
	rm -f *~ *.o *.csv $(BENCHMARK)

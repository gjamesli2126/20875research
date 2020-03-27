#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
include ../../../common/cuda.mk

all: bh

bh.o: bh.cu bh.h 
	$(NVCC) $(NVCC_OPTIONS) -c bh.cu

bh_block.o: bh_block.cu bh_block.h
	$(NVCC) $(NVCC_OPTIONS) -c bh_block.cu

bh_kernel_mem.o: bh_kernel_mem.cu bh_kernel_mem.cu
	$(NVCC) $(NVCC_OPTIONS) -c bh_kernel_mem.cu

bh_kernel.o: bh_kernel.cu bh_kernel.h
	$(NVCC) $(NVCC_OPTIONS) -c bh_kernel.cu

bh: bh.o bh_block.o bh_kernel.o bh_kernel_mem.o
	$(NVCC) -o bh bh.o bh_block.o bh_kernel.o bh_kernel_mem.o

clean:
	rm -f *.o bh *~



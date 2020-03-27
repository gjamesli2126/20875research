#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
include ../../../common/cuda.mk

all: nn

nn.o: nn.cu nn.h
	$(NVCC) $(NVCC_OPTIONS) -c nn.cu

gpu_tree.o: gpu_tree.cu nn.h ptrtab.h
	$(NVCC) $(NVCC_OPTIONS) -c gpu_tree.cu

kernel.o: kernel.cu nn.h
	$(NVCC) $(NVCC_OPTIONS) -c kernel.cu

ptrtab.o: ptrtab.cu ptrtab.h
	$(NVCC) $(NVCC_OPTIONS) -c ptrtab.cu

nn: nn.o ptrtab.o gpu_tree.o kernel.o
	$(NVCC) $(NVCC_LINK_OPTIONS) -o nn nn.o ptrtab.o gpu_tree.o kernel.o

clean:
	rm -f *.o *~ nn



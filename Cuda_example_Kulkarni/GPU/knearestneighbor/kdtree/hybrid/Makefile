#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
include ../../../common/cuda.mk

all: knn

knn: knn.o kdtree.o dev_memory.o gpu_tree.o kernel.o pre_kernel.o
	$(NVCC) $(NVCC_LINK_OPTIONS) -o knn knn.o kdtree.o gpu_tree.o dev_memory.o kernel.o pre_kernel.o

knn.o: knn.cu knn.h
	$(NVCC) $(NVCC_OPTIONS) -c knn.cu

kdtree.o: kdtree.cu knn.h
	$(NVCC) $(NVCC_OPTIONS) -c kdtree.cu

gpu_tree.o: gpu_tree.cu gpu_tree.h knn.h
	$(NVCC) $(NVCC_OPTIONS) -c gpu_tree.cu

dev_memory.o: dev_memory.cu knn.h
	$(NVCC) $(NVCC_OPTIONS) -c dev_memory.cu

kernel.o: kernel.cu knn.h knn_gpu.h knn_kernel_macros.inc
	$(NVCC) $(NVCC_OPTIONS) -c kernel.cu

pre_kernel.o: pre_kernel.cu knn.h knn_gpu.h knn_pre_kernel_macros.inc
	$(NVCC) $(NVCC_OPTIONS) -c pre_kernel.cu

clean:
	rm -f *.o knn *~



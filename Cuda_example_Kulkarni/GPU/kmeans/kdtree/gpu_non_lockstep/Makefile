#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and 
# * Purdue University. All Rights Reserved. See Copyright.txt
#*************************************************************************************************
include ../../../common/cuda.mk
#NVCC=g++

all: kmeans

kmeans: cpu_kernel.o gpu_kernel.o gpu_mem.o functions.o kmeans.o gpu_tree.o
	$(NVCC) $(NVCC_LINK_OPTIONS) -o kmeans cpu_kernel.o gpu_kernel.o gpu_mem.o functions.o kmeans.o gpu_tree.o

cpu_kernel.o: cpu_kernel.cu cpu_kernel.h
	$(NVCC) $(NVCC_OPTIONS) -c cpu_kernel.cu

gpu_kernel.o: gpu_kernel.cu gpu_kernel.h
	$(NVCC) $(NVCC_OPTIONS) -c gpu_kernel.cu

gpu_mem.o: gpu_mem.cu gpu_mem.h
	$(NVCC) $(NVCC_OPTIONS) -c gpu_mem.cu

gpu_tree.o: gpu_tree.cu gpu_tree.h
	$(NVCC) $(NVCC_OPTIONS) -c gpu_tree.cu

functions.o: functions.cu functions.h
	$(NVCC) $(NVCC_OPTIONS) -c functions.cu

kmeans.o: kmeans.cu common.h functions.h
	$(NVCC) $(NVCC_OPTIONS) -c kmeans.cu

clean:
	rm -rf *.o kmeans *~

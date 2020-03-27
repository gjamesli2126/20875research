#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
include ../../../common/cuda.mk
#NVCC=g++

all:knn

knn: knn_kernel.o knn_pre_kernel.o knn_functions.o knn.o knn_gpu_tree.o
	$(NVCC) $(NVCC_LINK_OPTIONS) -o knn knn_kernel.o knn_pre_kernel.o knn_functions.o knn.o knn_gpu_tree.o

knn_kernel.o: knn_kernel.cu knn_kernel.h
	$(NVCC) $(NVCC_OPTIONS) -c knn_kernel.cu

knn_pre_kernel.o: knn_pre_kernel.cu knn_pre_kernel.h
	$(NVCC) $(NVCC_OPTIONS) -c knn_pre_kernel.cu

knn_functions.o: knn_functions.cu knn_functions.h
	$(NVCC) $(NVCC_OPTIONS) -c knn_functions.cu

knn.o: knn.cu knn_common.h knn_functions.h
	$(NVCC) $(NVCC_OPTIONS) -c knn.cu
	
knn_gpu_tree.o: knn_gpu_tree.cu knn_gpu_tree.h
	$(NVCC) $(NVCC_OPTIONS) -c knn_gpu_tree.cu

clean:
	rm -rf *.o balltree *~

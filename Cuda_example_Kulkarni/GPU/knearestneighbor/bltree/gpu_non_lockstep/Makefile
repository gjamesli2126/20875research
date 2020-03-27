#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#***********************************************************************************************/
include ../../../common/cuda.mk
#NVCC=g++

all:balltree

balltree: bt_kernel.o bt_functions.o bt.o bt_gpu_tree.o
	$(NVCC) $(NVCC_LINK_OPTIONS) -o balltree bt_kernel.o bt_functions.o bt.o bt_gpu_tree.o

bt_kernel.o: bt_kernel.cu bt_kernel.h
	$(NVCC) $(NVCC_OPTIONS) -c bt_kernel.cu

bt_functions.o: bt_functions.cu bt_functions.h
	$(NVCC) $(NVCC_OPTIONS) -c bt_functions.cu

bt.o: bt.cu bt_common.h bt_functions.h
	$(NVCC) $(NVCC_OPTIONS) -c bt.cu
	
bt_gpu_tree.o: bt_gpu_tree.cu bt_gpu_tree.h
	$(NVCC) $(NVCC_OPTIONS) -c bt_gpu_tree.cu

clean:
	rm -rf *.o balltree *~

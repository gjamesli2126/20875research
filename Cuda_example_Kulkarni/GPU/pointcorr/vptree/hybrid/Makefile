include ../../../common/cuda.mk

all: pc

pc.o: pc.cu pc.h
	$(NVCC) $(NVCC_OPTIONS) -c pc.cu

gpu_tree.o: gpu_tree.cu pc.h ptrtab.h
	$(NVCC) $(NVCC_OPTIONS) -c gpu_tree.cu

kernel.o: kernel.cu pc.h
	$(NVCC) $(NVCC_OPTIONS) -c kernel.cu

ptrtab.o: ptrtab.cu ptrtab.h
	$(NVCC) $(NVCC_OPTIONS) -c ptrtab.cu

pc: pc.o ptrtab.o gpu_tree.o kernel.o
	$(NVCC) $(NVCC_LINK_OPTIONS) -o pc pc.o ptrtab.o gpu_tree.o kernel.o

clean:
	rm -f *.o *~ pc



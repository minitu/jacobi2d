all: jacobi2d halotest

jacobi2d: jacobi2d.o
	mpicxx -o $@ $^ -L$(CUDA_HOME)/lib64 -lcudart

jacobi2d.o: jacobi2d.cu
	nvcc -c $< -arch=sm_30 -Xptxas -dlcm=cg -I$(MPI_ROOT)/include -I$(HOME)/cub-1.8.0

halotest: halotest.o
	mpicxx -o $@ $^ #-L$(HOME)/sst-dumpi/install/lib -ldumpi

halotest.o: halotest.cpp
	mpicxx -c $<

clean:
	rm -rf jacobi2d halotest *.o

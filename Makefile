all: jacobi2d halotest

jacobi2d: main.o stencil.o
	mpicxx -o $@ $^ -L$(CUDA_DIR)/lib64 -lcudart

halotest: halotest.o
	mpicxx -o $@ $^ #-L$(HOME)/sst-dumpi/install/lib -ldumpi

halotest.o: halotest.cpp
	mpicxx -c $<

main.o: main.cpp stencil.h
	mpicxx -c $<

stencil.o: stencil.cu stencil.h
	nvcc -c $< -arch=sm_30 -Xptxas -dlcm=cg -I$(HOME)/cub-1.8.0

clean:
	rm -rf jacobi2d halotest *.o

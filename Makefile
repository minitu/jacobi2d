all: jacobi2d

jacobi2d: main.o stencil.o
	mpicxx -o $@ $^ -lcudart -L$(CUDA_DIR)/lib64

main.o: main.cpp stencil.h
	mpicxx -c $<

stencil.o: stencil.cu stencil.h
	nvcc -c $<

clean:
	rm -rf jacobi2d *.o

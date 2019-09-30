all: jacobi2d

jacobi2d: main.o stencil.o
	mpicxx -pthread -o $@ $^ -L$(CUDA_DIR)/lib64 -lcudart

main.o: main.cpp stencil.h
	mpicxx -DDEBUG -pthread -c $<

stencil.o: stencil.cu stencil.h
	nvcc -c $< -arch=compute_70 -code=sm_70

clean:
	rm -rf jacobi2d *.o

all: jacobi2d

jacobi2d: main.o stencil.o
	xlc++_r -pthread -o $@ $^ -L$(CUDA_DIR)/lib64 -lcudart -L/ccs/home/jchoi/liballprof-0.9/install/lib -lclog -L$(MPI_ROOT)/lib -lmpi_ibm

main.o: main.cpp stencil.h
	xlc++_r -pthread -c $<

stencil.o: stencil.cu stencil.h
	nvcc -c $< -arch=compute_70 -code=sm_70

clean:
	rm -rf jacobi2d *.o

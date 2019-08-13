all: jacobi2d

jacobi2d: main.cpp domain.h
	mpicxx -o $@ $<

clean:
	rm -rf jacobi2d

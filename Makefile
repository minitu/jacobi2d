all: jacobi2d

jacobi2d: main.cpp
	mpicxx -o $@ $<

clean:
	rm -rf jacobi2d

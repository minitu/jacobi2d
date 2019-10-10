#ifndef STENCIL_H_
#define STENCIL_H_

#define DEBUG 1
#define USE_GPU 1
#define N_TIMER 9
#define N_DUR (N_TIMER-1)

#define IND(x,y) (block_size+2)*(y)+(x)

void gpuSet(int rank);
bool gpuAllocate(double** a_old, double** a_new, double** sbuf_north, double** sbuf_south,
    double** sbuf_east, double** sbuf_west, double** rbuf_north, double** rbuf_south,
    double** rbuf_east, double** rbuf_west, double** tbuf_east, double** tbuf_west,
    int block_size, double** d_local_heat, double** h_local_heat);
void gpuRandInit(double* a_old, int block_size);
void gpuPackHalo(double* a_old, double* sbuf_north, double* sbuf_south,
    double* sbuf_east, double* sbuf_west, double* tbuf_east, double* tbuf_west,
    int block_size);
void gpuUnpackHalo(double* a_old, double* rbuf_north, double* rbuf_south,
    double* rbuf_east, double* rbuf_west, double* tbuf_east, double* tbuf_west,
    int block_size);
void gpuStencil(double* a_old, double* a_new, int block_size);
void gpuReduce(double* a_new, int block_size, double* d_local_heat, double* h_local_heat);

#endif // STENCIL_H_

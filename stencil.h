#ifndef STENCIL_H_
#define STENCIL_H_

#define DEBUG 1
#define USE_GPU 0
#define NONBLOCKING 1
#define N_TIMER 7
#define N_DUR (N_TIMER-1)

#if USE_GPU
#define IND(x,y) (n_pitch)*(y)+(x)
#else
#define IND(x,y) (bx+2)*(y)+(x)
#endif

bool gpuAllocate(double** a_old, double** a_new, size_t* pitch, double** sbuf_north,
    double** sbuf_south, double** sbuf_east, double** sbuf_west, double** rbuf_north,
    double** rbuf_south, double** rbuf_east, double** rbuf_west, int bx, int by,
    double** d_local_heat, double** h_local_heat);
void gpuRandInit(double* a_old, int bx, int by, size_t pitch);
void gpuPackHalo(double* a_old, int bx, int by, size_t pitch, double* sbuf_north,
    double* sbuf_south, double* sbuf_east, double* sbuf_west);
void gpuUnpackHalo(double* a_old, int bx, int by, size_t pitch, double* rbuf_north,
    double* rbuf_south, double* rbuf_east, double* rbuf_west);
void gpuStencil(double* a_old, double* a_new, int bx, int by, size_t pitch);
void gpuReduce(double* a_new, int bx, int by, size_t pitch, double* d_local_heat,
    double* h_local_heat);

#endif // STENCIL_H_

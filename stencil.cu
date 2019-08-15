#include <stdio.h>
#include <curand_kernel.h>
#include "stencil.h"

#define TILE_SIZE 16
#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
  }
}

__global__ void randInitKernel(double* a_old, int bx, int by, size_t pitch) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  curandState state;
  curand_init(IND(i,j), 0, 0, &state);

  if (i >= 1 && i <= bx && j >= 1 && j <= by) {
    a_old[IND(i,j)] = curand_uniform_double(&state) * 10;
  }
}

__global__ void stencilKernel(double* a_old, double* a_new, int bx, int by, size_t pitch) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= 1 && i <= bx && j >= 1 && j <= by) {
    a_new[IND(i,j)] = (a_old[IND(i,j)] + (a_old[IND(i-1,j)] + a_old[IND(i+1,j)] + a_old[IND(i,j-1)] + a_old[IND(i,j+1)]) * 0.25) * 0.5;
  }

  /* TODO Use shared memory
  int i = istart + threadIdx.x + blockDim.x*blockIdx.x;
  int j = jstart + threadIdx.y + blockDim.y*blockIdx.y;

  if (i < ifinish && j < jfinish) {
    __shared__ double shared_a_old[TILE_SIZE][TILE_SIZE];
    double center = a_old[j*(bx+2)+i];

    shared_a_old[threadIdx.x][threadIdx.y] = center;
    __syncthreads();

    // update my value based on the surrounding values
    a_new[j*(bx+2)+i] = (
      ((threadIdx.x > 1) ? shared_a_old[threadIdx.x-1][threadIdx.y] :
  a_old[j*(bx+2)+(i-1)]) +
      ((threadIdx.x < blockDim.x-1) ?
  shared_a_old[threadIdx.x+1][threadIdx.y] :
  a_old[j*(bx+2)+(i+1)]) +
      ((threadIdx.y > 1) ? shared_a_old[threadIdx.x][threadIdx.y-1] :
  a_old[(j-1)*(bx+2)+i]) +
      ((threadIdx.y < blockDim.y-1) ?
  shared_a_old[threadIdx.x][threadIdx.y+1] :
  a_old[(j+1)*(bx+2)+i]) +
      center) * 0.2;
  }
  */
}

bool gpuAllocate(double** a_old, double** a_new, size_t* pitch, double** sbuf_north,
    double** sbuf_south, double** sbuf_east, double** sbuf_west, double** rbuf_north,
    double** rbuf_south, double** rbuf_east, double** rbuf_west, int bx, int by) {
  // 2D pitched memory allocation for efficient access
  size_t second_pitch;
  gpuCheck(cudaMallocPitch(a_old, pitch, (bx+2) * sizeof(double), by));
  gpuCheck(cudaMallocPitch(a_new, &second_pitch, (bx+2) * sizeof(double), by));
  if (*pitch != second_pitch) {
    fprintf(stderr, "Pitches are different: %d, %d\n", *pitch, second_pitch);
    return false;
  }

  // Host communication buffers
  gpuCheck(cudaMallocHost(sbuf_north, bx * sizeof(double)));
  gpuCheck(cudaMallocHost(sbuf_south, bx * sizeof(double)));
  gpuCheck(cudaMallocHost(sbuf_east, by * sizeof(double)));
  gpuCheck(cudaMallocHost(sbuf_west, by * sizeof(double)));
  gpuCheck(cudaMallocHost(rbuf_north, bx * sizeof(double)));
  gpuCheck(cudaMallocHost(rbuf_south, bx * sizeof(double)));
  gpuCheck(cudaMallocHost(rbuf_east, by * sizeof(double)));
  gpuCheck(cudaMallocHost(rbuf_west, by * sizeof(double)));

  return true;
}

void gpuRandInit(double* a_old, int bx, int by, size_t pitch) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((bx + block_dim.x - 1) / block_dim.x, (by + block_dim.y - 1) / block_dim.y);
  randInitKernel<<<grid_dim, block_dim>>>(a_old, bx, by, pitch);
  gpuCheck(cudaPeekAtLastError());
}

void gpuPackHalo(double** a_old, size_t pitch, double** sbuf_north, double** sbuf_south,
    double** sbuf_east, double** sbuf_west) {

}

void gpuStencil(double* a_old, double* a_new, int bx, int by, size_t pitch) {
  // TODO: H2D transfer

  // Stencil kernel
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((bx + block_dim.x - 1) / block_dim.x, (by + block_dim.y - 1) / block_dim.y);
  stencilKernel<<<grid_dim, block_dim>>>(a_old, a_new, bx, by, pitch);
  gpuCheck(cudaPeekAtLastError());

  // TODO: Reduction kernel for local heat

  // TODO: D2H transfer
}

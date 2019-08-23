#include <stdio.h>
#include <curand_kernel.h>
#include "stencil.h"

#define SHARED_MEM 0
#define TILE_SIZE 16
#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
  }
}

__global__ void randInitKernel(double* a_old, int bx, int by, size_t n_pitch) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  curandState state;
  curand_init(IND(i,j), 0, 0, &state);

  if (i >= 1 && i <= bx && j >= 1 && j <= by) {
    a_old[IND(i,j)] = curand_uniform_double(&state) * 10;
  }
}

__global__ void stencilKernel(double* a_old, double* a_new, int bx, int by, size_t n_pitch) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
#if !SHARED_MEM
  if (i >= 1 && i <= bx && j >= 1 && j <= by) {
    a_new[IND(i,j)] = (a_old[IND(i,j)] + (a_old[IND(i-1,j)] + a_old[IND(i+1,j)] + a_old[IND(i,j-1)] + a_old[IND(i,j+1)]) * 0.25) * 0.5;
  }
#else
  __shared__ double s_a_old[TILE_SIZE][TILE_SIZE];

  if (i >= 1 && i <= bx && j >= 1 && j <= by) {
    double center = a_old[IND(i,j)];
    s_a_old[threadIdx.x][threadIdx.y] = center;

    __syncthreads();

    a_new[IND(i,j)] = (center + (((threadIdx.x > 0 && i > 1) ? s_a_old[threadIdx.x-1][threadIdx.y] : a_old[IND(i-1,j)])
      + ((threadIdx.x < blockDim.x-1 && i < bx) ? s_a_old[threadIdx.x+1][threadIdx.y] : a_old[IND(i+1,j)])
      + ((threadIdx.y > 0 && j > 1) ? s_a_old[threadIdx.x][threadIdx.y-1] : a_old[IND(i,j-1)])
      + ((threadIdx.y < blockDim.y-1 && j < by) ? s_a_old[threadIdx.x][threadIdx.y+1] : a_old[IND(i,j+1)])) * 0.25) * 0.5;
  }
#endif
}

__global__ void sumKernel(double* a_new, int bx, int by, size_t n_pitch, double* sum) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  __shared__ double shared_sum;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    shared_sum = 0.0;
  }

  __syncthreads();

  // Sum reduction within thread block
  if (i >= 1 && i <= bx && j >= 1 && j <= by) {
    atomicAdd(&shared_sum, a_new[IND(i,j)]);
  }

  __syncthreads();

  // Sum reduction across thread blocks
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(sum, shared_sum);
  }
}

bool gpuAllocate(double** a_old, double** a_new, size_t* pitch, double** sbuf_north,
    double** sbuf_south, double** sbuf_east, double** sbuf_west, double** rbuf_north,
    double** rbuf_south, double** rbuf_east, double** rbuf_west, int bx, int by,
    double** d_local_heat, double** h_local_heat) {
  // 2D pitched memory allocation for efficient access
  size_t second_pitch;
  gpuCheck(cudaMallocPitch(a_old, pitch, (bx+2) * sizeof(double), (by+2)));
  gpuCheck(cudaMallocPitch(a_new, &second_pitch, (bx+2) * sizeof(double), (by+2)));
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

  // Local heat
  gpuCheck(cudaMalloc(d_local_heat, sizeof(double)));
  gpuCheck(cudaMallocHost(h_local_heat, sizeof(double)));

  return true;
}

void gpuRandInit(double* a_old, int bx, int by, size_t pitch) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((bx + block_dim.x - 1) / block_dim.x, (by + block_dim.y - 1) / block_dim.y);
  size_t n_pitch = pitch / sizeof(double);
  randInitKernel<<<grid_dim, block_dim>>>(a_old, bx, by, n_pitch);
  gpuCheck(cudaPeekAtLastError());

  cudaDeviceSynchronize();
}

void gpuPackHalo(double* a_old, int bx, int by, size_t pitch, double* sbuf_north,
    double* sbuf_south, double* sbuf_east, double* sbuf_west) {
  size_t n_pitch = pitch / sizeof(double);
  gpuCheck(cudaMemcpy2D(sbuf_north, bx * sizeof(double), &a_old[IND(1,0)], pitch,
        bx * sizeof(double), 1, cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy2D(sbuf_south, bx * sizeof(double), &a_old[IND(1,by+1)], pitch,
        bx * sizeof(double), 1, cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy2D(sbuf_east, sizeof(double), &a_old[IND(bx+1,1)], pitch,
        sizeof(double), by, cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy2D(sbuf_west, sizeof(double), &a_old[IND(0,1)], pitch,
        sizeof(double), by, cudaMemcpyDeviceToHost));
}

void gpuUnpackHalo(double* a_old, int bx, int by, size_t pitch, double* rbuf_north,
    double* rbuf_south, double* rbuf_east, double* rbuf_west) {
  size_t n_pitch = pitch / sizeof(double);
  gpuCheck(cudaMemcpy2D(&a_old[IND(1,0)], pitch, rbuf_north, bx * sizeof(double),
        bx * sizeof(double), 1, cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy2D(&a_old[IND(1,by+1)], pitch, rbuf_south, bx * sizeof(double),
        bx * sizeof(double), 1, cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy2D(&a_old[IND(bx+1,1)], pitch, rbuf_east, sizeof(double),
        sizeof(double), by, cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy2D(&a_old[IND(0,1)], pitch, rbuf_west, sizeof(double),
        sizeof(double), by, cudaMemcpyHostToDevice));
}

void gpuStencil(double* a_old, double* a_new, int bx, int by, size_t pitch,
    double* d_local_heat, double* h_local_heat) {
  // Stencil kernel
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((bx + block_dim.x - 1) / block_dim.x, (by + block_dim.y - 1) / block_dim.y);
  size_t n_pitch = pitch / sizeof(double);
  stencilKernel<<<grid_dim, block_dim>>>(a_old, a_new, bx, by, n_pitch);
  gpuCheck(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  // Reduction kernel for local heat
  gpuCheck(cudaMemcpy(d_local_heat, h_local_heat, sizeof(double), cudaMemcpyHostToDevice));
  block_dim.x = 8; block_dim.y = 8; // Smaller block size for more concurrency
  grid_dim.x = (bx + block_dim.x - 1) / block_dim.x;
  grid_dim.y = (by + block_dim.y - 1) / block_dim.y;
  sumKernel<<<grid_dim, block_dim>>>(a_new, bx, by, n_pitch, d_local_heat);
  cudaDeviceSynchronize();
  gpuCheck(cudaMemcpy(h_local_heat, d_local_heat, sizeof(double), cudaMemcpyDeviceToHost));
}

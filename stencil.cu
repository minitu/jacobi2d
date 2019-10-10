#include <stdio.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include "stencil.h"

#define SHARED_MEM 0
#define TILE_SIZE 16
#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
  }
}

// GPU kernels
__global__ void randInitKernel(double* a_old, int block_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  curandState state;
  curand_init(IND(i,j), 0, 0, &state);

  if (i >= 1 && i <= block_size && j >= 1 && j <= block_size) {
    a_old[IND(i,j)] = curand_uniform_double(&state) * 10;
  }
  else if (i < block_size+2 && j < block_size+2) {
    a_old[IND(i,j)] = 0; // Halo area set to 0
  }
}

__global__ void packKernel(double* a_old, double* tbuf_east, double* tbuf_west, int block_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  // West halo
  if (i == 0 && j >= 1 && j <= block_size) {
    tbuf_west[j-1] = a_old[IND(i,j)];
  }

  // East halo
  if (i == (block_size+1) && j >= 1 && j <= block_size) {
    tbuf_east[j-1] = a_old[IND(i,j)];
  }
}

__global__ void unpackKernel(double* a_old, double* tbuf_east, double* tbuf_west, int block_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  // West halo
  if (i == 0 && j >= 1 && j <= block_size) {
    a_old[IND(i,j)] = tbuf_west[j-1];
  }

  // East halo
  if (i == (block_size+1) && j >= 1 && j <= block_size) {
    a_old[IND(i,j)] = tbuf_east[j-1];
  }
}

__global__ void stencilKernel(double* a_old, double* a_new, int block_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
#if !SHARED_MEM
  if (i >= 1 && i <= block_size && j >= 1 && j <= block_size) {
    a_new[IND(i,j)] = (a_old[IND(i,j)] + (a_old[IND(i-1,j)] + a_old[IND(i+1,j)] + a_old[IND(i,j-1)] + a_old[IND(i,j+1)]) * 0.25) * 0.5;
  }
#else
  __shared__ double s_a_old[TILE_SIZE][TILE_SIZE];

  if (i >= 1 && i <= block_size && j >= 1 && j <= block_size) {
    double center = a_old[IND(i,j)];
    s_a_old[threadIdx.x][threadIdx.y] = center;

    __syncthreads();

    a_new[IND(i,j)] = (center + (((threadIdx.x > 0 && i > 1) ? s_a_old[threadIdx.x-1][threadIdx.y] : a_old[IND(i-1,j)])
      + ((threadIdx.x < blockDim.x-1 && i < block_size) ? s_a_old[threadIdx.x+1][threadIdx.y] : a_old[IND(i+1,j)])
      + ((threadIdx.y > 0 && j > 1) ? s_a_old[threadIdx.x][threadIdx.y-1] : a_old[IND(i,j-1)])
      + ((threadIdx.y < blockDim.y-1 && j < block_size) ? s_a_old[threadIdx.x][threadIdx.y+1] : a_old[IND(i,j+1)])) * 0.25) * 0.5;
  }
#endif
}

/*
__global__ void sumKernel(double* a_new, int block_size, double* sum) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  __shared__ double shared_sum;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    shared_sum = 0.0;
  }

  __syncthreads();

  // Sum reduction within thread block
  if (i >= 1 && i <= block_size && j >= 1 && j <= block_size) {
    atomicAdd(&shared_sum, a_new[IND(i,j)]);
  }

  __syncthreads();

  // Sum reduction across thread blocks
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(sum, shared_sum);
  }
}
*/

// Host-side functions
void gpuSet(int rank) {
  int n_devices;
  gpuCheck(cudaGetDeviceCount(&n_devices));
  if (rank == 0) {
    printf("# of visible GPU devices per rank: %d\n", n_devices);
  }
  gpuCheck(cudaSetDevice(rank % n_devices));
}

bool gpuAllocate(double** a_old, double** a_new, double** sbuf_north, double** sbuf_south,
    double** sbuf_east, double** sbuf_west, double** rbuf_north, double** rbuf_south,
    double** rbuf_east, double** rbuf_west, double** tbuf_east, double** tbuf_west,
    int block_size, double** d_local_heat, double** h_local_heat) {
  // Block buffers
  gpuCheck(cudaMalloc(a_old, (block_size+2) * (block_size+2) * sizeof(double)));
  gpuCheck(cudaMalloc(a_new, (block_size+2) * (block_size+2) * sizeof(double)));

  // Host communication buffers
  // XXX: Using cudaMallocHost significantly speeds up MPI halo communication
  /*
  gpuCheck(cudaMallocHost(sbuf_north, block_size * sizeof(double)));
  gpuCheck(cudaMallocHost(sbuf_south, block_size * sizeof(double)));
  gpuCheck(cudaMallocHost(sbuf_east, block_size * sizeof(double)));
  gpuCheck(cudaMallocHost(sbuf_west, block_size * sizeof(double)));
  gpuCheck(cudaMallocHost(rbuf_north, block_size * sizeof(double)));
  gpuCheck(cudaMallocHost(rbuf_south, block_size * sizeof(double)));
  gpuCheck(cudaMallocHost(rbuf_east, block_size * sizeof(double)));
  gpuCheck(cudaMallocHost(rbuf_west, block_size * sizeof(double)));
  */
  *sbuf_north = (double*)malloc(block_size * sizeof(double));
  *sbuf_south = (double*)malloc(block_size * sizeof(double));
  *sbuf_east = (double*)malloc(block_size * sizeof(double));
  *sbuf_west = (double*)malloc(block_size * sizeof(double));
  *rbuf_north = (double*)malloc(block_size * sizeof(double));
  *rbuf_south = (double*)malloc(block_size * sizeof(double));
  *rbuf_east = (double*)malloc(block_size * sizeof(double));
  *rbuf_west = (double*)malloc(block_size * sizeof(double));

  // Temporary device buffers for packing & unpacking
  gpuCheck(cudaMalloc(tbuf_east, block_size * sizeof(double)));
  gpuCheck(cudaMalloc(tbuf_west, block_size * sizeof(double)));

  // Local heat
  gpuCheck(cudaMalloc(d_local_heat, sizeof(double)));
  gpuCheck(cudaMallocHost(h_local_heat, sizeof(double)));

  return true;
}

void gpuRandInit(double* a_old, int block_size) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(((block_size + 2) + block_dim.x - 1) / block_dim.x,
      ((block_size + 2) + block_dim.y - 1) / block_dim.y);
  randInitKernel<<<grid_dim, block_dim>>>(a_old, block_size);
  gpuCheck(cudaPeekAtLastError());

  cudaDeviceSynchronize();
}

void gpuPackHalo(double* a_old, double* sbuf_north, double* sbuf_south,
    double* sbuf_east, double* sbuf_west, double* tbuf_east, double* tbuf_west,
    int block_size) {
  // Move east & west halo data to contiguous device buffers
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(((block_size + 2) + block_dim.x - 1) / block_dim.x,
      ((block_size + 2) + block_dim.y - 1) / block_dim.y);
  packKernel<<<grid_dim, block_dim>>>(a_old, tbuf_east, tbuf_west, block_size);
  gpuCheck(cudaPeekAtLastError());

  cudaDeviceSynchronize();

  // Move device halo data to host
  gpuCheck(cudaMemcpy(sbuf_north, &a_old[IND(1,0)], block_size * sizeof(double),
        cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy(sbuf_south, &a_old[IND(1,block_size+1)], block_size * sizeof(double),
        cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy(sbuf_east, tbuf_east, block_size * sizeof(double),
        cudaMemcpyDeviceToHost));
  gpuCheck(cudaMemcpy(sbuf_west, tbuf_west, block_size * sizeof(double),
        cudaMemcpyDeviceToHost));
}

void gpuUnpackHalo(double* a_old, double* rbuf_north, double* rbuf_south,
    double* rbuf_east, double* rbuf_west, double* tbuf_east, double* tbuf_west,
    int block_size) {
  // Move host halo data to device
  gpuCheck(cudaMemcpy(&a_old[IND(1,0)], rbuf_north, block_size * sizeof(double),
        cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(&a_old[IND(1,block_size+1)], rbuf_south, block_size * sizeof(double),
        cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(tbuf_east, rbuf_east, block_size * sizeof(double),
        cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(tbuf_west, rbuf_west, block_size * sizeof(double),
        cudaMemcpyHostToDevice));

  // Move received east & west halo data to the right places
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(((block_size + 2) + block_dim.x - 1) / block_dim.x,
      ((block_size + 2) + block_dim.y - 1) / block_dim.y);
  unpackKernel<<<grid_dim, block_dim>>>(a_old, tbuf_east, tbuf_west, block_size);
  gpuCheck(cudaPeekAtLastError());

  cudaDeviceSynchronize();
}

void gpuStencil(double* a_old, double* a_new, int block_size) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(((block_size + 2) + block_dim.x - 1) / block_dim.x, ((block_size + 2) + block_dim.y - 1) / block_dim.y);
  stencilKernel<<<grid_dim, block_dim>>>(a_old, a_new, block_size);
  gpuCheck(cudaPeekAtLastError());

  cudaDeviceSynchronize();
}

void gpuReduce(double* a_new, int block_size, double* d_local_heat, double* h_local_heat) {
  // Use CUB to perform a sum reduction on all values
  void* temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, a_new, d_local_heat, (block_size + 2) * (block_size + 2));

  gpuCheck(cudaMalloc(&temp_storage, temp_storage_bytes));

  cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, a_new, d_local_heat, (block_size + 2) * (block_size + 2));
  gpuCheck(cudaPeekAtLastError());

  cudaDeviceSynchronize();

  // Copy reduced heat to host
  gpuCheck(cudaMemcpy(h_local_heat, d_local_heat, sizeof(double), cudaMemcpyDeviceToHost));
}

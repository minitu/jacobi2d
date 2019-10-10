/*
 * Author: Jaemin Choi <jaemin@acm.org>
 *
 * Adapted from SC17 MPI Tutorial code examples
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <time.h>
#include <float.h>
#include <assert.h>

#include <curand_kernel.h>
#include <cub/cub.cuh>

#define DEBUG 1
#define N_TIMER 12
#define N_DUR N_TIMER

#define IND(x,y) (block_size+2)*(y)+(x)

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

int main(int argc, char** argv) {
  // Command line arguments
  int c;
  int block_size = 128;
  int n_iters = 100;

  while ((c = getopt(argc, argv, "s:i:")) != -1) {
    switch (c) {
      case 's':
        block_size = atoi(optarg);
        break;
      case 'i':
        n_iters = atoi(optarg);
        break;
      default:
        abort();
    }
  }

  // Initialize MPI
  MPI_Init(NULL, NULL);

  int rank, world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Determine dimensions of rank grid
  int pdims[2] = {0, 0};
  MPI_Dims_create(world_size, 2, pdims);
  int px = pdims[0]; int py = pdims[1];

  // Create Cartesian topology of ranks
  int periods[2] = {0, 0};
  MPI_Comm topo_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, pdims, periods, 0, &topo_comm);

  // Get x & y coordinates of my rank
  int pcoords[2];
  MPI_Cart_coords(topo_comm, rank, 2, pcoords);

  int north, south, east, west;
  MPI_Cart_shift(topo_comm, 0, 1, &west, &east);
  MPI_Cart_shift(topo_comm, 1, 1, &north, &south);
  int n_neighbors = 0;
  if (north >= 0) n_neighbors++;
  if (south >= 0) n_neighbors++;
  if (east >= 0) n_neighbors++;
  if (west >= 0) n_neighbors++;

#if DEBUG
  if (rank == 0) {
    printf("Rank grid: (%d, %d)\n", px, py);
    printf("Block size: %d x %d\n", block_size, block_size);
    printf("Entire domain: %d x %d\n", block_size * px, block_size * py);
  }
  /*
  int rx = pcoords[0]; int ry = pcoords[1];
  printf("[Rank %d] (%d, %d), N: %d, E: %d, S: %d, W: %d\n",
      rank, rx, ry, north, east, south, west);
  */
#endif

  // Map rank to GPU (necessary for non-jsrun systems)
  int n_devices;
  gpuCheck(cudaGetDeviceCount(&n_devices));
  if (rank == 0) {
    printf("# of visible GPU devices per rank: %d\n", n_devices);
  }
  gpuCheck(cudaSetDevice(rank % n_devices));

  // Allocate temperature data & communication buffers
  double* a_old; double* a_new;
  double* sbuf_north; double* sbuf_south;
  double* sbuf_east; double* sbuf_west;
  double* rbuf_north; double* rbuf_south;
  double* rbuf_east; double* rbuf_west;
  double* tbuf_east; double* tbuf_west;
  double* d_local_heat; double* h_local_heat;

  // Block buffers
  gpuCheck(cudaMalloc(&a_old, (block_size+2) * (block_size+2) * sizeof(double)));
  gpuCheck(cudaMalloc(&a_new, (block_size+2) * (block_size+2) * sizeof(double)));

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
  sbuf_north = (double*)malloc(block_size * sizeof(double));
  sbuf_south = (double*)malloc(block_size * sizeof(double));
  sbuf_east = (double*)malloc(block_size * sizeof(double));
  sbuf_west = (double*)malloc(block_size * sizeof(double));
  rbuf_north = (double*)malloc(block_size * sizeof(double));
  rbuf_south = (double*)malloc(block_size * sizeof(double));
  rbuf_east = (double*)malloc(block_size * sizeof(double));
  rbuf_west = (double*)malloc(block_size * sizeof(double));

  // Temporary device buffers for packing & unpacking
  gpuCheck(cudaMalloc(&tbuf_east, block_size * sizeof(double)));
  gpuCheck(cudaMalloc(&tbuf_west, block_size * sizeof(double)));

  // Local heat
  gpuCheck(cudaMalloc(&d_local_heat, sizeof(double)));
  gpuCheck(cudaMallocHost(&h_local_heat, sizeof(double)));

  if (!(sbuf_north && sbuf_south && sbuf_east && sbuf_west && rbuf_north
        && rbuf_south && rbuf_east && rbuf_west)) {
    if (rank == 0) printf("Memory allocation failed!\n");
    MPI_Abort(topo_comm, MPI_ERR_OTHER);
  }

  // Block and grid dimensions for GPU kernels
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim(((block_size + 2) + block_dim.x - 1) / block_dim.x,
      ((block_size + 2) + block_dim.y - 1) / block_dim.y);

  // Randomly initialize temperature
  randInitKernel<<<grid_dim, block_dim>>>(a_old, block_size);
  gpuCheck(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  // Heat of system
  double global_heat_old = 0.0;
  double global_heat_new = 0.0;

  // Timers
  double local_times[N_TIMER];
  double local_durations[N_DUR];
  double local_durations_sum[N_DUR];
  double local_durations_min[N_DUR];
  double global_durations[N_DUR*world_size];
  for (int i = 0; i < N_DUR; i++) {
    local_durations_sum[i] = 0;
    local_durations_min[i] = DBL_MAX;
  }

  // Main iteration loop
  for (int iter = 1; iter <= n_iters; iter++) {
    local_times[0] = MPI_Wtime();

    // Pack halo data
    packKernel<<<grid_dim, block_dim>>>(a_old, tbuf_east, tbuf_west, block_size);
    gpuCheck(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    local_times[1] = MPI_Wtime();

    // Move device halo data to host
    gpuCheck(cudaMemcpy(sbuf_north, &a_old[IND(1,0)], block_size * sizeof(double),
          cudaMemcpyDeviceToHost));
    gpuCheck(cudaMemcpy(sbuf_south, &a_old[IND(1,block_size+1)], block_size * sizeof(double),
          cudaMemcpyDeviceToHost));
    gpuCheck(cudaMemcpy(sbuf_east, tbuf_east, block_size * sizeof(double),
          cudaMemcpyDeviceToHost));
    gpuCheck(cudaMemcpy(sbuf_west, tbuf_west, block_size * sizeof(double),
          cudaMemcpyDeviceToHost));

    local_times[2] = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    local_times[3] = MPI_Wtime();

    MPI_Request reqs[8];

    // Send & receive halos
    int req_i = 0;
    if (north >= 0) MPI_Irecv(rbuf_north, block_size, MPI_DOUBLE, north, 9, topo_comm, &reqs[req_i++]);
    if (south >= 0) MPI_Irecv(rbuf_south, block_size, MPI_DOUBLE, south, 9, topo_comm, &reqs[req_i++]);
    if (east >= 0) MPI_Irecv(rbuf_east, block_size, MPI_DOUBLE, east, 9, topo_comm, &reqs[req_i++]);
    if (west >= 0) MPI_Irecv(rbuf_west, block_size, MPI_DOUBLE, west, 9, topo_comm, &reqs[req_i++]);

    if (north >= 0) MPI_Isend(sbuf_north, block_size, MPI_DOUBLE, north, 9, topo_comm, &reqs[req_i++]);
    if (south >= 0) MPI_Isend(sbuf_south, block_size, MPI_DOUBLE, south, 9, topo_comm, &reqs[req_i++]);
    if (east >= 0) MPI_Isend(sbuf_east, block_size, MPI_DOUBLE, east, 9, topo_comm, &reqs[req_i++]);
    if (west >= 0) MPI_Isend(sbuf_west, block_size, MPI_DOUBLE, west, 9, topo_comm, &reqs[req_i++]);

    assert(n_neighbors*2 == req_i);
    MPI_Waitall(n_neighbors*2, reqs, MPI_STATUSES_IGNORE);

    local_times[4] = MPI_Wtime();

    // Move received halo data to device
    gpuCheck(cudaMemcpy(&a_old[IND(1,0)], rbuf_north, block_size * sizeof(double),
          cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(&a_old[IND(1,block_size+1)], rbuf_south, block_size * sizeof(double),
          cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(tbuf_east, rbuf_east, block_size * sizeof(double),
          cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(tbuf_west, rbuf_west, block_size * sizeof(double),
          cudaMemcpyHostToDevice));

    local_times[5] = MPI_Wtime();

    // Move received east & west halo data to the right places
    unpackKernel<<<grid_dim, block_dim>>>(a_old, tbuf_east, tbuf_west, block_size);
    gpuCheck(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    local_times[6] = MPI_Wtime();

    // Update temperatures
    stencilKernel<<<grid_dim, block_dim>>>(a_old, a_new, block_size);
    gpuCheck(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    local_times[7] = MPI_Wtime();

    // Use CUB to perform a sum reduction on all values
    void* temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, a_new, d_local_heat, (block_size + 2) * (block_size + 2));
    gpuCheck(cudaMalloc(&temp_storage, temp_storage_bytes));
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, a_new, d_local_heat, (block_size + 2) * (block_size + 2));
    gpuCheck(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    local_times[8] = MPI_Wtime();

    // Copy reduced heat to host
    gpuCheck(cudaMemcpy(h_local_heat, d_local_heat, sizeof(double), cudaMemcpyDeviceToHost));

    local_times[9] = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    local_times[10] = MPI_Wtime();

    // Sum up all local heat values
    MPI_Allreduce(h_local_heat, &global_heat_new, 1, MPI_DOUBLE, MPI_SUM, topo_comm);

    local_times[11] = MPI_Wtime();

    // Check difference in global heat
    /*
    if (rank == 0) {
      printf("[%03d] old: %.3lf, new: %.3lf, diff: %.3lf\n", iter,
          global_heat_old, global_heat_new, global_heat_new - global_heat_old);
    }
    */

    // Swap arrays and values
    double* a_tmp;
    a_tmp = a_new; a_new = a_old; a_old = a_tmp;
    global_heat_old = global_heat_new;

    // Calculate durations
    for (int i = 0; i < N_DUR-1; i++) {
      local_durations[i] = local_times[i+1] - local_times[i];
    }
    local_durations[N_DUR-1] = local_times[N_TIMER-1] - local_times[0];

    for (int i = 0; i < N_DUR; i++) {
      // Don't include first iteration in the sum
      if (iter > 0) {
        local_durations_sum[i] += local_durations[i];
      }
      // Minimum time
      if (local_durations[i] < local_durations_min[i]) {
        local_durations_min[i] = local_durations[i];
      }
    }

    // Gather times
    MPI_Gather(local_durations, N_DUR, MPI_DOUBLE, global_durations, N_DUR, MPI_DOUBLE, 0, topo_comm);

    // Print per-iteration times
    /*
    if (rank == 0) {
      for (int j = 0; j < world_size; j++) {
        double duration_sum = 0;
        for (int i = 0; i < N_DUR-1; i++) {
          duration_sum += global_durations[N_DUR*j+i];
        }
        double overhead = global_durations[N_DUR*j+N_DUR-1] - duration_sum;

        printf("[%03d,%03d] Pack: %.3lf, D2H: %.3lf, Barrier: %.3lf, Halo: %.3lf, "
            "H2D: %.3lf, Unpack: %.3lf, Stencil: %.3lf, Reduce: %.3lf, R-D2H: %.3lf, "
            "Barrier: %.3lf, Allreduce: %.3lf, Overhead: %.3lf, Iter: %.3lf\n", iter, j,
            global_durations[N_DUR*j] * 1000000, global_durations[N_DUR*j+1] * 1000000,
            global_durations[N_DUR*j+2] * 1000000, global_durations[N_DUR*j+3] * 1000000,
            global_durations[N_DUR*j+4] * 1000000, global_durations[N_DUR*j+5] * 1000000,
            global_durations[N_DUR*j+6] * 1000000, global_durations[N_DUR*j+7] * 1000000,
            global_durations[N_DUR*j+8] * 1000000, global_durations[N_DUR*j+9] * 1000000,
            global_durations[N_DUR*j+10] * 1000000, overhead * 1000000,
            global_durations[N_DUR*j+11] * 1000000);
      }
    }
    */
  }

  // Print average times
  MPI_Gather(local_durations_sum, N_DUR, MPI_DOUBLE, global_durations, N_DUR, MPI_DOUBLE, 0, topo_comm);
  if (rank == 0) {
    for (int j = 0; j < world_size; j++) {
      double duration_sum = 0;
      for (int i = 0; i < N_DUR-1; i++) {
        duration_sum += global_durations[N_DUR*j+i];
      }
      double overhead = global_durations[N_DUR*j+N_DUR-1] - duration_sum;

      for (int i = 0; i < N_DUR; i++) {
        global_durations[N_DUR*j+i] /= (n_iters-1); // Don't include first iteration times
      }
      overhead /= (n_iters-1);

      printf("[average,%03d] Pack: %.3lf, D2H: %.3lf, Barrier: %.3lf, Halo: %.3lf, "
          "H2D: %.3lf, Unpack: %.3lf, Stencil: %.3lf, Reduce: %.3lf, R-D2H: %.3lf, "
          "Barrier: %.3lf, Allreduce: %.3lf, Overhead: %.3lf, Iter: %.3lf\n", j,
          global_durations[N_DUR*j] * 1000000, global_durations[N_DUR*j+1] * 1000000,
          global_durations[N_DUR*j+2] * 1000000, global_durations[N_DUR*j+3] * 1000000,
          global_durations[N_DUR*j+4] * 1000000, global_durations[N_DUR*j+5] * 1000000,
          global_durations[N_DUR*j+6] * 1000000, global_durations[N_DUR*j+7] * 1000000,
          global_durations[N_DUR*j+8] * 1000000, global_durations[N_DUR*j+9] * 1000000,
          global_durations[N_DUR*j+10] * 1000000, overhead * 1000000,
          global_durations[N_DUR*j+11] * 1000000);
    }
  }

  // Print minimum times
  MPI_Gather(local_durations_min, N_DUR, MPI_DOUBLE, global_durations, N_DUR, MPI_DOUBLE, 0, topo_comm);
  if (rank == 0) {
    for (int j = 0; j < world_size; j++) {
      printf("[minimum,%03d] Pack: %.3lf, D2H: %.3lf, Barrier: %.3lf, Halo: %.3lf, "
          "H2D: %.3lf, Unpack: %.3lf, Stencil: %.3lf, Reduce: %.3lf, R-D2H: %.3lf, "
          "Barrier: %.3lf, Allreduce: %.3lf, Iter: %.3lf\n", j,
          global_durations[N_DUR*j] * 1000000, global_durations[N_DUR*j+1] * 1000000,
          global_durations[N_DUR*j+2] * 1000000, global_durations[N_DUR*j+3] * 1000000,
          global_durations[N_DUR*j+4] * 1000000, global_durations[N_DUR*j+5] * 1000000,
          global_durations[N_DUR*j+6] * 1000000, global_durations[N_DUR*j+7] * 1000000,
          global_durations[N_DUR*j+8] * 1000000, global_durations[N_DUR*j+9] * 1000000,
          global_durations[N_DUR*j+10] * 1000000, global_durations[N_DUR*j+11] * 1000000);
    }
  }

  // Print maxmium of the minimum times across ranks
  if (rank == 0) {
    double max_times[N_DUR];
    for (int i = 0; i < N_DUR; i++) {
      max_times[i] = 0;
      for (int j = 0; j < world_size; j++) {
        if (global_durations[N_DUR*j+i] * 1000000 > max_times[i]) {
          max_times[i] = global_durations[N_DUR*j+i] * 1000000;
        }
      }
    }

    printf("[final] Max Pack: %.3lf, D2H: %.3lf, Barrier: %.3lf, Halo: %.3lf, "
        "H2D: %.3lf, Unpack: %.3lf, Stencil: %.3lf, Reduce: %.3lf, R-D2H: %.3lf, "
        "Barrier: %.3lf, Allreduce: %.3lf, Iter: %.3lf\n",
        max_times[0], max_times[1], max_times[2], max_times[3], max_times[4],
        max_times[5], max_times[6], max_times[7], max_times[8], max_times[9],
        max_times[10], max_times[11]);
  }

  // Finalize MPI
  MPI_Finalize();

  return 0;
}

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
#include "stencil.h"

int main(int argc, char** argv) {
  // Command line arguments
  int c;
  int domain_size = 128;
  int n_iters = 100;

  while ((c = getopt(argc, argv, "s:i:")) != -1) {
    switch (c) {
      case 's':
        domain_size = atoi(optarg);
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
  int rx = pcoords[0]; int ry = pcoords[1];

  int north, south, east, west;
  MPI_Cart_shift(topo_comm, 0, 1, &west, &east);
  MPI_Cart_shift(topo_comm, 1, 1, &north, &south);

#if DEBUG
  if (rank == 0) {
    printf("Rank grid: (%d, %d)\n", px, py);
  }
  printf("[Rank %d] (%d, %d), N: %d, E: %d, S: %d, W: %d\n",
      rank, rx, ry, north, east, south, west);
#endif

  // Domain decomposition
  int bx = domain_size / px;
  int by = domain_size / py;
#if DEBUG
  if (rank == 0) {
    printf("bx: %d, by: %d\n", bx, by);
  }
#endif

  // Allocate temperature data & communication buffers
  double* a_old; double* a_new;
  size_t pitch;
  double* sbuf_north; double* sbuf_south;
  double* sbuf_east; double* sbuf_west;
  double* rbuf_north; double* rbuf_south;
  double* rbuf_east; double* rbuf_west;
  double* d_local_heat; double* h_local_heat;
  bool allocate_success;
#if USE_GPU
  allocate_success = gpuAllocate(&a_old, &a_new, &pitch, &sbuf_north, &sbuf_south,
      &sbuf_east, &sbuf_west, &rbuf_north, &rbuf_south, &rbuf_east, &rbuf_west, bx, by,
      &d_local_heat, &h_local_heat);
#else
  a_old = (double*)malloc((bx+2) * (by+2) * sizeof(double));
  a_new = (double*)malloc((bx+2) * (by+2) * sizeof(double));
  sbuf_north = (double*)malloc(bx * sizeof(double));
  sbuf_south = (double*)malloc(bx * sizeof(double));
  sbuf_east = (double*)malloc(by * sizeof(double));
  sbuf_west = (double*)malloc(by * sizeof(double));
  rbuf_north = (double*)malloc(bx * sizeof(double));
  rbuf_south = (double*)malloc(bx * sizeof(double));
  rbuf_east = (double*)malloc(by * sizeof(double));
  rbuf_west = (double*)malloc(by * sizeof(double));
  h_local_heat = (double*)malloc(sizeof(double));
  allocate_success = a_old && a_new && sbuf_north && sbuf_south && sbuf_east
    && sbuf_west && rbuf_north && rbuf_south && rbuf_east && rbuf_west && h_local_heat;
#endif
  if (!allocate_success) {
    if (rank == 0) printf("Memory allocation failed!\n");
    MPI_Abort(topo_comm, MPI_ERR_OTHER);
  }

  // Randomly initialize temperature
#if USE_GPU
  gpuRandInit(a_old, bx, by, pitch);
#else
  srand(time(NULL));
  for (int j = 1; j <= by; j++) {
    for (int i = 1; i <= bx; i++) {
      a_old[IND(i,j)] = rand() % 100;
    }
  }
#endif

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
#if USE_GPU
    gpuPackHalo(a_old, bx, by, pitch, sbuf_north, sbuf_south, sbuf_east, sbuf_west);
#else
    for (int i = 1; i <= bx; i++) {
      sbuf_north[i-1] = a_old[IND(i,0)];
    }
    for (int i = 1; i <= bx; i++) {
      sbuf_south[i-1] = a_old[IND(i,by+1)];
    }
    for (int j = 1; j <= by; j++) {
      sbuf_east[j-1] = a_old[IND(bx+1,j)];
    }
    for (int j = 1; j <= by; j++) {
      sbuf_west[j-1] = a_old[IND(0,j)];
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    local_times[1] = MPI_Wtime();

    MPI_Request reqs[8];

    // Send & receive halos
#if NONBLOCKING
    MPI_Irecv(rbuf_north, bx, MPI_DOUBLE, north, 9, topo_comm, &reqs[4]);
    MPI_Irecv(rbuf_south, bx, MPI_DOUBLE, south, 9, topo_comm, &reqs[5]);
    MPI_Irecv(rbuf_east, by, MPI_DOUBLE, east, 9, topo_comm, &reqs[6]);
    MPI_Irecv(rbuf_west, by, MPI_DOUBLE, west, 9, topo_comm, &reqs[7]);

    MPI_Isend(sbuf_north, bx, MPI_DOUBLE, north, 9, topo_comm, &reqs[0]);
    MPI_Isend(sbuf_south, bx, MPI_DOUBLE, south, 9, topo_comm, &reqs[1]);
    MPI_Isend(sbuf_east, by, MPI_DOUBLE, east, 9, topo_comm, &reqs[2]);
    MPI_Isend(sbuf_west, by, MPI_DOUBLE, west, 9, topo_comm, &reqs[3]);

    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
#else
    // TODO
#endif

    local_times[2] = MPI_Wtime();

#if USE_GPU
    // Unpack halo data
    gpuUnpackHalo(a_old, bx, by, pitch, rbuf_north, rbuf_south, rbuf_east, rbuf_west);
#else
    for (int i = 1; i <= bx; i++) {
      a_old[IND(i,0)] = rbuf_north[i-1];
    }
    for (int i = 1; i <= bx; i++) {
      a_old[IND(i,by+1)] = rbuf_south[i-1];
    }
    for (int j = 1; j <= by; j++) {
      a_old[IND(bx+1,j)] = rbuf_east[j-1];
    }
    for (int j = 1; j <= by; j++) {
      a_old[IND(0,j)] = rbuf_west[j-1];
    }
#endif

    local_times[3] = MPI_Wtime();

    // Update temperatures
#if USE_GPU
    gpuStencil(a_old, a_new, bx, by, pitch);
#else
    for (int j = 1; j <= by; j++) {
      for (int i = 1; i <= bx; i++) {
        a_new[IND(i,j)] = (a_old[IND(i,j)] + (a_old[IND(i-1,j)] + a_old[IND(i+1,j)]
              + a_old[IND(i,j-1)] + a_old[IND(i,j+1)]) * 0.25) * 0.5;
      }
    }
#endif

    local_times[4] = MPI_Wtime();

    // Reduce local heat
    *h_local_heat = 0.0;
#if USE_GPU
    gpuReduce(a_new, bx, by, pitch, d_local_heat, h_local_heat);
#else
    for (int j = 1; j <= by; j++) {
      for (int i = 1; i <= bx; i++) {
        *h_local_heat += a_new[IND(i,j)];
      }
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    local_times[5] = MPI_Wtime();

    // Sum up all local heat values
    MPI_Allreduce(h_local_heat, &global_heat_new, 1, MPI_DOUBLE, MPI_SUM, topo_comm);

    local_times[6] = MPI_Wtime();

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
    for (int i = 0; i < N_DUR; i++) {
      local_durations[i] = local_times[i+1] - local_times[i];
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
    if (rank == 0) {
      for (int j = 0; j < world_size; j++) {
        double iter_time = 0;
        for (int i = 0; i < N_DUR; i++) {
          iter_time += global_durations[N_DUR*j+i];
        }

        printf("[%03d,%03d] Pack: %.3lf, MPI Halo: %.3lf, Unpack: %.3lf, Stencil: %.3lf,"
            " Reduce: %.3lf, MPI Allreduce: %.3lf, Iter: %.3lf\n", iter, j,
            global_durations[N_DUR*j] * 1000000, global_durations[N_DUR*j+1] * 1000000,
            global_durations[N_DUR*j+2] * 1000000, global_durations[N_DUR*j+3] * 1000000,
            global_durations[N_DUR*j+4] * 1000000, global_durations[N_DUR*j+5] * 1000000,
            iter_time * 1000000);
      }
    }
  }

  /*
  // Print average times
  MPI_Gather(local_durations_sum, N_DUR, MPI_DOUBLE, global_durations, N_DUR, MPI_DOUBLE, 0, topo_comm);
  if (rank == 0) {
    for (int j = 0; j < world_size; j++) {
      double total_time = 0;
      for (int i = 0; i < N_DUR; i++) {
        global_durations[N_DUR*j+i] /= (n_iters-1); // Don't include first iteration times
        total_time += global_durations[N_DUR*j+i];
      }

      printf("[average,%03d] Pack: %.3lf, MPI Halo: %.3lf, Unpack: %.3lf, Stencil: %.3lf,"
          " Reduce; %.3lf, MPI Allreduce: %.3lf, Iter: %.3lf\n", j,
          global_durations[N_DUR*j] * 1000000, global_durations[N_DUR*j+1] * 1000000,
          global_durations[N_DUR*j+2] * 1000000, global_durations[N_DUR*j+3] * 1000000,
          global_durations[N_DUR*j+4] * 1000000, global_durations[N_DUR*j+5] * 1000000,
          total_time * 1000000);
    }
  }
  */

  // Print average times
  MPI_Gather(local_durations_min, N_DUR, MPI_DOUBLE, global_durations, N_DUR, MPI_DOUBLE, 0, topo_comm);
  if (rank == 0) {
    for (int j = 0; j < world_size; j++) {
      double total_time = 0;
      for (int i = 0; i < N_DUR; i++) {
        total_time += global_durations[N_DUR*j+i];
      }

      printf("[minimum,%03d] Pack: %.3lf, MPI Halo: %.3lf, Unpack: %.3lf, Stencil: %.3lf,"
          " Reduce; %.3lf, MPI Allreduce: %.3lf, Iter: %.3lf\n", j,
          global_durations[N_DUR*j] * 1000000, global_durations[N_DUR*j+1] * 1000000,
          global_durations[N_DUR*j+2] * 1000000, global_durations[N_DUR*j+3] * 1000000,
          global_durations[N_DUR*j+4] * 1000000, global_durations[N_DUR*j+5] * 1000000,
          total_time * 1000000);
    }
  }


  // Finalize MPI
  MPI_Finalize();

  return 0;
}

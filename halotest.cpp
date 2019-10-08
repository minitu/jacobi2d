/*
 * Author: Jaemin Choi <jaemin@acm.org>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <time.h>
#include <float.h>
#include <assert.h>

#define DEBUG 1
#define N_TIMER 3
#define N_DUR (N_TIMER-1)
#define TRACE_MODE 1
#define K_MODE 1

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
  int rx = pcoords[0]; int ry = pcoords[1];

  int north, south, east, west;
  MPI_Cart_shift(topo_comm, 0, 1, &west, &east);
  MPI_Cart_shift(topo_comm, 1, 1, &north, &south);
  int n_neighbors = 0;
  if (north >= 0) n_neighbors++;
  if (south >= 0) n_neighbors++;
  if (east >= 0) n_neighbors++;
  if (west >= 0) n_neighbors++;

#if K_MODE
  // Measures K (capital K)
  int* k_values = NULL;
  int k_value = 0;
  int off_node = 0;
  int node_size = 6;

  int my_node = rank / node_size;
  int north_node = (north >= 0) ? (north / node_size) : my_node;
  int south_node = (south >= 0) ? (south / node_size) : my_node;
  int east_node = (east >= 0) ? (east / node_size) : my_node;
  int west_node = (west >= 0) ? (west / node_size) : my_node;

  if (north_node != my_node) off_node++;
  if (south_node != my_node) off_node++;
  if (east_node != my_node) off_node++;
  if (west_node != my_node) off_node++;

  // Create intra-node communicator
  MPI_Comm node_comm;
  MPI_Comm_split(MPI_COMM_WORLD, my_node, rank, &node_comm);

  // Calculate k values by all-reducing within node
  MPI_Allreduce(&off_node, &k_value, 1, MPI_INT, MPI_SUM, node_comm);

  // Gather all ranks' k values and print
  if (rank == 0) k_values = (int*)malloc(sizeof(int)*world_size);

  MPI_Gather(&k_value, 1, MPI_INT, k_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int i = 0; i < world_size; i++) {
      printf("[Rank %d] k_value: %d\n", i, k_values[i]);
    }
  }

  if (rank == 0) free(k_values);

  MPI_Finalize();
  return 0;
#endif

#if DEBUG
  if (rank == 0) {
    printf("Rank grid: (%d, %d)\n", px, py);
    printf("Block size: %d x %d\n", block_size, block_size);
    printf("Entire domain: %d x %d\n", block_size * px, block_size * py);
  }
  printf("[Rank %d] (%d, %d), N: %d, E: %d, S: %d, W: %d\n",
      rank, rx, ry, north, east, south, west);
#endif

  // Allocate communication buffers
  double* sbuf_north; double* sbuf_south;
  double* sbuf_east; double* sbuf_west;
  double* rbuf_north; double* rbuf_south;
  double* rbuf_east; double* rbuf_west;
  bool allocate_success;

  sbuf_north = (double*)malloc(block_size * sizeof(double));
  sbuf_south = (double*)malloc(block_size * sizeof(double));
  sbuf_east = (double*)malloc(block_size * sizeof(double));
  sbuf_west = (double*)malloc(block_size * sizeof(double));
  rbuf_north = (double*)malloc(block_size * sizeof(double));
  rbuf_south = (double*)malloc(block_size * sizeof(double));
  rbuf_east = (double*)malloc(block_size * sizeof(double));
  rbuf_west = (double*)malloc(block_size * sizeof(double));

  allocate_success = sbuf_north && sbuf_south && sbuf_east && sbuf_west
    && rbuf_north && rbuf_south && rbuf_east && rbuf_west;

  if (!allocate_success) {
    if (rank == 0) printf("Memory allocation failed!\n");
    MPI_Abort(topo_comm, MPI_ERR_OTHER);
  }

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
    for (int i = 1; i <= block_size; i++) {
      sbuf_north[i-1] = iter * i;
    }
    for (int i = 1; i <= block_size; i++) {
      sbuf_south[i-1] = iter * i;
    }
    for (int j = 1; j <= block_size; j++) {
      sbuf_east[j-1] = iter * j;
    }
    for (int j = 1; j <= block_size; j++) {
      sbuf_west[j-1] = iter * j;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    local_times[1] = MPI_Wtime();

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

    local_times[2] = MPI_Wtime();

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

#if !TRACE_MODE
    // Gather times
    MPI_Gather(local_durations, N_DUR, MPI_DOUBLE, global_durations, N_DUR, MPI_DOUBLE, 0, topo_comm);

    // Print per-iteration times
    if (rank == 0) {
      for (int j = 0; j < world_size; j++) {
        double iter_time = 0;
        for (int i = 0; i < N_DUR; i++) {
          iter_time += global_durations[N_DUR*j+i];
        }

        printf("[%03d,%03d] Pack: %.3lf, MPI Halo: %.3lf, Iter: %.3lf\n", iter, j,
            global_durations[N_DUR*j] * 1000000, global_durations[N_DUR*j+1] * 1000000,
            iter_time * 1000000);
      }
    }
#endif
  }

#if !TRACE_MODE
  // Print average times
  MPI_Gather(local_durations_sum, N_DUR, MPI_DOUBLE, global_durations, N_DUR, MPI_DOUBLE, 0, topo_comm);
  if (rank == 0) {
    for (int j = 0; j < world_size; j++) {
      double total_time = 0;
      for (int i = 0; i < N_DUR; i++) {
        global_durations[N_DUR*j+i] /= (n_iters-1); // Don't include first iteration times
        total_time += global_durations[N_DUR*j+i];
      }

      printf("[average,%03d] Pack: %.3lf, MPI Halo: %.3lf, Iter: %.3lf\n", j,
          global_durations[N_DUR*j] * 1000000, global_durations[N_DUR*j+1] * 1000000,
          total_time * 1000000);
    }
  }

  // Print minimum times
  MPI_Gather(local_durations_min, N_DUR, MPI_DOUBLE, global_durations, N_DUR, MPI_DOUBLE, 0, topo_comm);
  if (rank == 0) {
    for (int j = 0; j < world_size; j++) {
      double total_time = 0;
      for (int i = 0; i < N_DUR; i++) {
        total_time += global_durations[N_DUR*j+i];
      }

      printf("[minimum,%03d] Pack: %.3lf, MPI Halo: %.3lf, Iter: %.3lf\n", j,
          global_durations[N_DUR*j] * 1000000, global_durations[N_DUR*j+1] * 1000000,
          total_time * 1000000);
    }
  }

  // Print maximum of the minimum halo times across ranks
  if (rank == 0) {
    double halo_max = 0;
    for (int j = 0; j < world_size; j++) {
      if (global_durations[N_DUR*j+1] > halo_max) {
        halo_max = global_durations[N_DUR*j+1];
      }
    }

    printf("[final] Max MPI Halo: %.3lf\n", halo_max * 1000000);
  }
#endif

  // Finalize MPI
  MPI_Finalize();

  return 0;
}

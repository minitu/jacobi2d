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

#ifdef DEBUG
  if (rank == 0) {
    printf("Rank grid: (%d, %d)\n", px, py);
  }
  printf("[Rank %d] (%d, %d), N: %d, E: %d, S: %d, W: %d\n",
      rank, rx, ry, north, east, south, west);
#endif

  // Domain decomposition
  int bx = domain_size / px;
  int by = domain_size / py;

  // Allocate temperature data & communication buffers
  double* a_old; double* a_new;
  size_t pitch;
  double* sbuf_north; double* sbuf_south;
  double* sbuf_east; double* sbuf_west;
  double* rbuf_north; double* rbuf_south;
  double* rbuf_east; double* rbuf_west;
  bool allocate_success = gpuAllocate(&a_old, &a_new, &pitch, &sbuf_north, &sbuf_south,
      &sbuf_east, &sbuf_west, &rbuf_north, &rbuf_south, &rbuf_east, &rbuf_west, bx, by);
  if (!allocate_success) MPI_Abort(topo_comm, MPI_ERR_OTHER);

  // Randomly initialize temperature
  gpuRandInit(a_old, bx, by, pitch);

  // Heat of system
  double local_heat = 0.0;
  double global_heat_old = 0.0;
  double global_heat_new = 0.0;

  // Main iteration loop
  for (int iter = 1; iter <= n_iters; iter++) {
    // Pack halo data
    for (int i = 0; i < bx; i++) sbuf_north[i] = h_a_old[IND(1+i,1)];
    for (int i = 0; i < bx; i++) sbuf_south[i] = h_a_old[IND(1+i,by)];
    for (int i = 0; i < by; i++) sbuf_east[i] = h_a_old[IND(bx,1+i)];
    for (int i = 0; i < by; i++) sbuf_west[i] = h_a_old[IND(1,1+i)];

    MPI_Request reqs[8];

    // Send & receive halos
    MPI_Isend(sbuf_north, bx, MPI_DOUBLE, north, 9, topo_comm, &reqs[0]);
    MPI_Isend(sbuf_south, bx, MPI_DOUBLE, south, 9, topo_comm, &reqs[1]);
    MPI_Isend(sbuf_east, by, MPI_DOUBLE, east, 9, topo_comm, &reqs[2]);
    MPI_Isend(sbuf_west, by, MPI_DOUBLE, west, 9, topo_comm, &reqs[3]);

    MPI_Irecv(rbuf_north, bx, MPI_DOUBLE, north, 9, topo_comm, &reqs[4]);
    MPI_Irecv(rbuf_south, bx, MPI_DOUBLE, south, 9, topo_comm, &reqs[5]);
    MPI_Irecv(rbuf_east, by, MPI_DOUBLE, east, 9, topo_comm, &reqs[6]);
    MPI_Irecv(rbuf_west, by, MPI_DOUBLE, west, 9, topo_comm, &reqs[7]);

    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

    // Unpack halo data
    for (int i = 0; i < bx; i++) h_a_old[IND(1+i,0)] = rbuf_north[i];
    for (int i = 0; i < bx; i++) h_a_old[IND(1+i,by+1)] = rbuf_south[i];
    for (int i = 0; i < by; i++) h_a_old[IND(bx+1,1+i)] = rbuf_east[i];
    for (int i = 0; i < by; i++) h_a_old[IND(0,1+i)] = rbuf_west[i];

    // Update temperatures
    local_heat = 0.0;
    gpuStencil(h_a_old, h_a_new, d_a_old, d_a_new, bx, by, 1);

    // Sum up all local heat values
    MPI_Allreduce(&local_heat, &global_heat_new, 1, MPI_DOUBLE, MPI_SUM, topo_comm);

    // Check difference in global heat
    if (rank == 0) {
      printf("[%03d] old: %.3lf, new: %.3lf, diff: %.3lf\n", iter,
          global_heat_old, global_heat_new, global_heat_new - global_heat_old);
    }

    // Swap arrays and values
    double* a_tmp;
    a_tmp = h_a_new; h_a_new = h_a_old; h_a_old = a_tmp;
    global_heat_old = global_heat_new;
  }

  // Finalize MPI
  MPI_Finalize();

  return 0;
}

#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include "domain.h"

int main(int argc, char** argv) {
  // Command line arguments
  int c;
  int domain_size = 128;
  bool weak_scaling = true;

  while ((c = getopt(argc, argv, "s:w")) != -1) {
    switch (c) {
      case 's':
        domain_size = atoi(optarg);
        break;
      case 'w':
        weak_scaling = true;
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

  // Create domain & decompose using recursive bisection
  Domain sim_domain(domain_size);
  sim_domain.decompose(rank, world_size);

  MPI_Finalize();

  return 0;
}

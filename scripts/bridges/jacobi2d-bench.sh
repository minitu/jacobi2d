#!/bin/bash
#SBATCH -p GPU
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:p100:2
#SBATCH --time=01:00:00
#SBATCH --job-name=jacobi2d-n6

date

cd /home/jchoi157/jacobi2d

ranks=6
block_size=16384

export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

for iter in 1 2 3
do
  echo "Running iteration $iter"
  mpiexec -print-rank-map -n $ranks -ppn 2 -genv I_MPI_DEBUG=5 ./jacobi2d-b -s $block_size -i 100 > jacobi2d-n"$ranks"-"$iter".out
done

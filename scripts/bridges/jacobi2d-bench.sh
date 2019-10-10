#!/bin/bash
#SBATCH -p GPU
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:p100:2
#SBATCH --time=00:30:00
#SBATCH --job-name=jacobi2d-n4

date

cd /g/g90/choi18/jacobi2d

block_size=16384

export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

for iter in 1 2 3
do
  echo "Running iteration $iter"
  mpiexec -print-rank-map -n $SLURM_NTASKS -genv I_MPI_DEBUG=5 ./jacobi2d-b -s $block_size -i 100 > jacobi2d-n"$SLURM_NTASKS"-"$iter".out
done

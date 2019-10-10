#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:p100:4
#SBATCH --ntasks-per-node=28
#SBATCH --nodes=8
#SBATCH --job-name="jacobi2d-p100-n32"
#SBATCH -t 00:00:10

date

cd /home/jchoi157/jacobi2d

ranks=32
block_size=16384

for iter in 1 2 3
do
  echo "Running iteration $iter"
  mpiexec -N 4 ./jacobi2d-b -s $block_size -i 100 > jacobi2d-p100-n"$ranks"-"$iter".out
done

#!/bin/bash
#BSUB -P csc357
#BSUB -W 0:10
#BSUB -nnodes 4
#BSUB -J jacobi2d-n24

date

cd $MEMBERWORK/csc357/jacobi2d

ranks=24
block_size=16384

for iter in 1 2 3
do
  echo "Running iteration $iter"
  jsrun -n $ranks -a 1 -c 1 -g 1 -K 3 -r 6 ./jacobi2d-b -s $block_size -i 100 > jacobi2d-n"$ranks"-"$iter".out
done

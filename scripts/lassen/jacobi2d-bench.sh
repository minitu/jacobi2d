#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -J jacobi2d-n4

date

cd /g/g90/choi18/jacobi2d

ranks=4
block_size=16384

for iter in 1 2 3
do
  echo "Running iteration $iter"
  jsrun -n $ranks -a 1 -c 1 -g 1 -K 2 -r 4 ./jacobi2d-b -s $block_size -i 100 > jacobi2d-n"$ranks"-"$iter".out
done

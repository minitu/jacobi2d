#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 512
#BSUB -J ht-n2048-s16384

date

cd /g/g90/choi18/jacobi2d

ranks=2048
block_size=16384

for iter in 1 2 3
do
  echo "Running iteration $iter"
  jsrun -n $ranks -a 1 -c 1 -K 2 -r 4 ./halotest -s $block_size -i 1000 > n"$ranks"-s"$block_size"-"$iter".out
done

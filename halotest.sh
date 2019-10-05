#!/bin/bash
#BSUB -P csc357
#BSUB -W 0:10
#BSUB -nnodes 256
#BSUB -J ht-n1536-s256

date

cd $MEMBERWORK/csc357/jacobi2d

ranks=1536
block_size=256

for iter in 1 2 3
do
  echo "Running iteration $iter"
  jsrun -n $ranks -a 1 -c 1 -K 3 -r 6 ./halotest -s $block_size -i 1000 > n"$ranks"-s"$block_size"-"$iter".out
done

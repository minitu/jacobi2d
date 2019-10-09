#!/bin/bash
#BSUB -P csc357
#BSUB -W 0:10
#BSUB -nnodes 256
#BSUB -J ht-n1536-s16384

date

cd $MEMBERWORK/csc357/jacobi2d

ranks=1536
block_size=16384

jsrun -n $ranks -a 1 -c 1 -K 3 -r 6 ./halotest-k -s $block_size -i 100 &> k-n"$ranks".out

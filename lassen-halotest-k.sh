#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 256
#BSUB -J ht-n1024

date

cd /g/g90/choi18/jacobi2d

ranks=1024
block_size=16384

jsrun -n $ranks -a 1 -c 1 -K 2 -r 4 ./halotest-k -s $block_size -i 100 &> k-n"$ranks".out

#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 256
#BSUB -J ht-d-n1024

date

cd /g/g90/choi18/jacobi2d

ranks=1024
block_size=16384

export LD_LIBRARY_PATH=/g/g90/choi18/sst-dumpi/install/lib:$LD_LIBRARY_PATH
jsrun -n $ranks -a 1 -c 1 -K 2 -r 4 ./halotest-d -s $block_size -i 100

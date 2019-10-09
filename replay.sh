#!/bin/bash

dir="summit-maxrate"
block_size=16384

#for rank in 6
for rank in 6 12 24 48 96 192 384 768 1536
do
  echo "Running $rank ranks"
  jsrun -n20 -a1 -c1 -K10 -r20 $HOME/work/codes-dumpi/build/src/network-workloads/model-net-mpi-replay --sync=3 --disable_compute=0 --workload_type="dumpi" --workload_file=/ccs/home/jchoi/work/jacobi2d/dumpi/s"$block_size"/n"$rank"- --num_net_traces="$rank" --lp-io-dir="$dir"/s"$block_size"/n"$rank" -- "$dir"/summit-replay.conf &> "$dir"/s"$block_size"/n"$rank".out
done

#!/bin/bash

machine="summit"
dir="$machine-replay"

#for rank in 6
for rank in 6 12 24 48 96 192 384 768 1536
do
  echo "Running $rank ranks"
  jsrun -n20 -a1 -c1 -K10 -r20 $HPM_PATH/codes/build/src/network-workloads/model-net-mpi-replay --sync=3 --disable_compute=1 --workload_type="dumpi" --workload_file=$HPM_PATH/apps/jacobi2d/summit-dumpi/s16384/n"$rank"- --num_net_traces="$rank" --lp-io-dir="$dir"/n"$rank" -- $HPM_PATH/conf/$machine/replay.conf &> "$dir"/n"$rank".out
done

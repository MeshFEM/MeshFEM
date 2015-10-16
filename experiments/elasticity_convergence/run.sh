#!/usr/bin/env zsh
# usage: ./run.sh degree poisson ratio
# e.g: ./run.sh 2 0.45
scriptdir=$(dirname $(readlink -f $0))
degree=$1
poisson=$2
mkdir -p sin/deg_$degree/poisson_$poisson
for res in {2..256..2}; do
    $MeshFEM/Simulate_cli <($MeshFEM/tools/grid ${res}x${res} -t -m'0,0' -M'1,1' /dev/stdout) -m <($scriptdir/material.sh $poisson) -b $scriptdir/sin_top.bc -d$degree -o sin/deg_$degree/poisson_$poisson/$res.msh
done

#!/usr/bin/env zsh
# usage: ./run.sh degree poisson
# e.g: ./run.sh 2 0.45
scriptdir=$(dirname $0)
poisson=$1
degree=$2

$MeshFEM/Simulate_cli mesh.msh -b $scriptdir/cantilever_2D.bc -m <($scriptdir/material.sh $poisson) -Do sim.msh

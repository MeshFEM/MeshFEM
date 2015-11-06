#!/usr/bin/env zsh
# usage: ./run.sh poisson degree
# e.g: ./run.sh 0.45 2
scriptdir=$(dirname $0)
poisson=$1
degree=$2

$MeshFEM/Simulate_cli mesh.msh -b $scriptdir/cantilever_2D.bc -m <($scriptdir/material.sh $poisson) -d$degree -Do sim.msh

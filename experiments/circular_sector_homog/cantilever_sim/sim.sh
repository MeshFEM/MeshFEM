#!/usr/bin/env zsh
# usage: ./run.sh poisson degree
# e.g: ./run.sh 0.45 2
scriptdir=$(dirname $0)
poisson=$1
degree=$2

# Mathematica only supports 2D obj--use this for both.
$MeshFEM/mesh_convert mesh.msh mesh.obj
mathematicaDir=$MeshFEM/experiments/mathematica_compare
$mathematicaDir/RemoveZComponentObj.sh mesh.obj

$MeshFEM/Simulate_cli mesh.obj -b $scriptdir/cantilever_2D.bc -m <($scriptdir/material.sh $poisson) -d$degree -Do sim.msh

# Run the Mathematica cantilever implementation
MathematicaScript -script $mathematicaDir/cantilever_elasticity.m mesh.obj $degree 200 $poisson

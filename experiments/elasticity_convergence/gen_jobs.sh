#!/usr/bin/env zsh
# generate the jobs to run on hpc
dir=$SCRATCH/FEM_convergence/elasticity
mkdir -p $dir $dir/jobs
for deg in {1,2}; do
for poisson in {-45..45..5}; do
    poisson=$(printf "%0.2f" $(($poisson / 100.0)))
    echo $poisson
    echo "cd $dir; $MeshFEM/experiments/elasticity_convergence/run.sh $deg $poisson" | create_pbs_from_stdin.sh "${deg}_$poisson" 2 8 1 0 > $dir/jobs/${deg}_$poisson.pbs
done
done

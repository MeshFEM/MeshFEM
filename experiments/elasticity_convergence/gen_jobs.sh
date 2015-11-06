#!/usr/bin/env zsh
# generate the jobs to run on hpc
dir=$SCRATCH/FEM_convergence/elasticity
mkdir -p $dir $dir/bc_jobs
for bc in {linear_top,sin_top,sin_2top,sin_3top,sin_4top,sin_5top,sin_full}; do
for deg in {1,2}; do
for poisson in {-49..49..1}; do
    poisson=$(printf "%0.2f" $(($poisson / 100.0)))
    echo "cd $dir; $MeshFEM/experiments/elasticity_convergence/run.sh $deg $poisson $bc" | create_pbs_from_stdin.sh "${bc}_${deg}_$poisson" 2 16 4 0 > $dir/bc_jobs/${bc}_${deg}_$poisson.pbs
done
done
echo $bc
done

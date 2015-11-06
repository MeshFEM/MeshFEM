#!/usr/bin/env zsh
# generate the jobs to run on hpc
dir=$SCRATCH/circ_sector_cantilever
mkdir -p $dir $dir/jobs
for poisson in {-48..48..4}; do
    poisson=$(printf "%0.2f" $(($poisson / 100.0)))
    for deg in {1,2}; do
        for skip in {-1..11}; do
            echo "mkdir -p $dir/skip_$skip/poisson_$poisson/deg_$deg; cd $dir/skip_$skip/poisson_$poisson/deg_$deg; python $MeshFEM/experiments/circular_sector_homog/cantilever_sim/convergence.py $skip $poisson $deg" | create_pbs_from_stdin.sh "s${skip}_${poisson}_${deg}" 2 16 1 0 > $dir/jobs/s${skip}_${poisson}_${deg}.pbs
        done
    done
done

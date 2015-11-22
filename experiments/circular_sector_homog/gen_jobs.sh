#!/usr/bin/env zsh
# generate the jobs to run on hpc
dir=$SCRATCH/circular_sector_homog
mkdir -p $dir $dir/jobs
for deg in {1,2}; do
for skip in {0..11}; do
    echo "mkdir -p $dir/skip_$skip/deg_$deg; cd $dir/skip_$skip/deg_$deg; python $MeshFEM/experiments/circular_sector_homog/convergence.py $skip $deg" | create_pbs_from_stdin.sh "${skip}_${deg}" 2 16 1 0 > $dir/jobs/${skip}_${deg}.pbs
done
done

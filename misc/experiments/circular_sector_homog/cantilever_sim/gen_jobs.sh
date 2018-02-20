#!/usr/bin/env zsh
# generate the jobs to run on hpc
dir=$SCRATCH/circ_sector_cantilever
mkdir -p $dir $dir/jobs
count=0
summaryCount=0
commandFile=$dir/jobs/compute_commands.sh
summCommandFile=$dir/jobs/summary_commands.sh
rm -f $commandFile $summCommandFile
for poisson in {-48..48..4}; do
    poisson=$(printf "%0.2f" $(($poisson / 100.0)))
    for skip in {-1..11}; do
        for deg in {1,2}; do
            echo "mkdir -p $dir/skip_$skip/poisson_$poisson/deg_$deg; cd $dir/skip_$skip/poisson_$poisson/deg_$deg; python $MeshFEM/experiments/circular_sector_homog/cantilever_sim/convergence.py $skip $poisson $deg" >> $commandFile
            count=$(($count + 1))
        done
        echo "python $MeshFEM/experiments/circular_sector_homog/cantilever_sim/summarize.py $SCRATCH/circ_sector_cantilever $skip $poisson" >> $summCommandFile
        summaryCount=$(($summaryCount + 1))
    done
done
cat > $dir/jobs/compute_array_job.pbs <<END
#!/bin/bash

###-----PBS Directives Start-----###

#PBS -V
#PBS -S /bin/bash
#PBS -N cs_cantilever_sim
#PBS -l nodes=1:ppn=2
#PBS -l walltime=1:00:00
#PBS -l mem=8GB
#PBS -M fjp234@nyu.edu
#PBS -m a
#PBS -e localhost:\${PBS_O_WORKDIR}/\${PBS_JOBNAME}.e\${PBS_JOBID}
#PBS -o localhost:\${PBS_O_WORKDIR}/\${PBS_JOBNAME}.o\${PBS_JOBID}
#PBS -t 1-$count

###-----PBS Directives End-----###
cd $dir/jobs
/bin/bash -c "\$(head -n\$PBS_ARRAYID compute_commands.sh | tail -n1)"
END
cat > $dir/jobs/summary_array_job.pbs <<END
#!/bin/bash

###-----PBS Directives Start-----###

#PBS -V
#PBS -S /bin/bash
#PBS -N summarize
#PBS -l nodes=1:ppn=1
#PBS -l walltime=0:10:00
#PBS -l mem=2GB
#PBS -M fjp234@nyu.edu
#PBS -m a
#PBS -e localhost:\${PBS_O_WORKDIR}/\${PBS_JOBNAME}.e\${PBS_JOBID}
#PBS -o localhost:\${PBS_O_WORKDIR}/\${PBS_JOBNAME}.o\${PBS_JOBID}
#PBS -t 1-$summaryCount

###-----PBS Directives End-----###
cd $dir/jobs
/bin/bash -c "\$(head -n\$PBS_ARRAYID summary_commands.sh | tail -n1)"
END

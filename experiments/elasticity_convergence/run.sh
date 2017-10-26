#!/usr/bin/env zsh
# usage: ./run.sh degree poisson ratio
# e.g: ./run.sh 2 0.45 sin_top
scriptdir=$(dirname $(readlink -f $0))
degree=$1
poisson=$2
bc=$3

dir=$bc/poisson_$poisson/deg_$degree
mkdir -p $dir
for res in {2..256..2}; do
    $MeshFEM/Simulate_cli <($MeshFEM/tools/grid ${res}x${res} -t -m'0,0' -M'1,1' /dev/stdout) -m <($scriptdir/material.sh $poisson) -b $scriptdir/$bc.bc -d$degree -o $dir/$res.msh -D "$dir/$res.bin"
    pushd $dir;
    echo -e $res\\t$(matlab -r "disp(condest_spd(upper_to_full(read_sparse_matrix_binary('$res.bin')))); exit" | tail -n2) >> condest.txt
    rm $res.bin
    popd
done
$scriptdir/sample_point.sh $dir 0.75,0.75
done

#!/usr/bin/env zsh
if [[ $ARGC != 2 ]]; then
    echo "Usage: sample_point.sh bc_name point";
    exit -1;
fi
point=$2
for p in {linear,quadratic}; do
    echo "Sampling $p..."
    ( for i in $1/$p/*.msh; do
        res=$(basename $i .msh)
        len=$(../../mesh_convert -i $i | grep "Min edge length" | cut -f2)
        echo "$res\t$len\t$($MeshFEM/tools/msh_processor $i -e 'u' --sample "$point")"
    done ) | tee $1/$p/$point.txt
done

#!/usr/bin/env zsh
if [[ $ARGC != 2 ]]; then
    echo "Usage: sample_point.sh directory point";
    exit -1;
fi
directory=$1
point=$2
( for i in $directory/*.msh; do
    res=$(basename $i .msh)
    len=$($MeshFEM/mesh_convert -i $i | grep "Min edge length" | cut -f2)
    echo "$res\t$len\t$($MeshFEM/tools/msh_processor $i -e 'u' --sample "$point" --dup --norm --applyAll --print | tr '\n' '\t')"
done ) | sort -n > $directory/$point.txt

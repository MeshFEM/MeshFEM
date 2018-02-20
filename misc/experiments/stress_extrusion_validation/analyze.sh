for i in /scratch/fjp234/extruded_sims_{2x2,250GB}/*.msh; do
    $MeshFEM/tools/msh_processor $i -e 'stress' -l --maxMag > $i.maxstress.txt
    $MeshFEM/tools/msh_processor $i -g 'volume' > $i.vol.txt
    stress=$($MeshFEM/tools/msh_processor $i -e 'stress' -l --maxMag --maxMag)
    echo -e "$i\t$stress"
done

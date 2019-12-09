#!/usr/bin/env zsh
for bc in results/*(/); do
    echo $bc
    ./compute_errors.py $bc
    for p in $bc/*(/); do
        p=$(basename $p)
        gnuplot -e "mesh='square'; bc_name='$bc'; poisson='$p'; error_type=0;" plot.gpi
        gnuplot -e "mesh='square'; bc_name='$bc'; poisson='$p'; error_type=1;" plot.gpi
        gnuplot -e "mesh='square'; bc_name='$bc'; poisson='$p'; error_type=2;" plot.gpi
        gnuplot -e "mesh='square'; bc_name='$bc'; poisson='$p';"               plot_condest.gpi
    done
done

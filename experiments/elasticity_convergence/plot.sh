#!/usr/bin/env zsh
for bc in {sin_top,linear_top}; do
for p in $bc/*(/); do
    p=$(basename $p)
       exact_ux=$(tail -n1 $bc/$p/deg_2/0.75,0.75.txt | cut -f3)
       exact_uy=$(tail -n1 $bc/$p/deg_2/0.75,0.75.txt | cut -f4)
    exact_unorm=$(tail -n1 $bc/$p/deg_2/0.75,0.75.txt | cut -f5)
    gnuplot -e "mesh='square'; bc_name='$bc'; poisson='$p'; error_type=0; exact='$exact_ux'"    plot.gpi
    gnuplot -e "mesh='square'; bc_name='$bc'; poisson='$p'; error_type=1; exact='$exact_uy'"    plot.gpi
    gnuplot -e "mesh='square'; bc_name='$bc'; poisson='$p'; error_type=2; exact='$exact_unorm'" plot.gpi
done
done

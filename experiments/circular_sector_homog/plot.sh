for i in results/skip_*; do
    for mod in {0..3}; do
        gnuplot -e "run_dir='$i'; modulus=$mod" plot_modulus_convergence.gpi
    done
    for ij in {0..2}; do
        gnuplot -e "run_dir='$i'; fluctuation=$ij" plot_maxstresses.gpi
    done
done

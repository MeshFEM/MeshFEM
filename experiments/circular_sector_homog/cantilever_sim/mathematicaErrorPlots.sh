mkdir -p mathematica_error_plots
for msh in {0..49}; do
    gnuplot -e "msh='$msh'" plotMathematicaErrors.gpi
done

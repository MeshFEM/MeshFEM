for i in results/skip_*/poisson_*; do
    nu=${i#*poisson_}
    cornerAngle=$(head -n1 $i/deg_1.txt | cut -f2)
    echo "nu: $nu, cornerAngle: $cornerAngle"
    for sample in {0..2}; do
        for comp in {x,y,norm}; do
            gnuplot -e "data_source='< ./errorData.py $i \"u_$comp[$sample]\"'; name='u at sample point $sample ($comp)'; title_suffix=' (nu = $nu, mesh corner angle $cornerAngle)'; png_path='$i/u_$comp.$sample.err.png'" ../plot_relerrors.gpi
            gnuplot -e "data_source='< ./errorData.py $i \"mathematica u_$comp[$sample]\"'; name='u at sample point $sample ($comp)'; title_suffix=' (nu = $nu, mesh corner angle $cornerAngle, Mathematica)'; png_path='$i/mathematica.u_$comp.$sample.err.png'" ../plot_relerrors.gpi
        done
    done
done

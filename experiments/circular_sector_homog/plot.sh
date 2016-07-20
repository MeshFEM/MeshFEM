for i in results/skip_*; do
    cornerAngle=$(head -n1 $i/deg_1.txt | cut -f2)
    for mod in {Ex,Ey,nu_yx,mu_xy}; do
        gnuplot -e "data_source='< ./errorData.py $i $mod'; name='$mod'; title_suffix=' (Mesh corner angle $cornerAngle)'; png_path='$i/$mod.png'" plot_relerrors.gpi
    done
    for ij in {0..2}; do
        for sample in {0..1}; do
            for comp in {x,y,norm}; do
                gnuplot -e "data_source='< ./errorData.py $i \"w${ij}_$comp[$sample]\"'; name='fluctuation displacement $ij at sample point $sample ($comp)'; title_suffix=' (Mesh corner angle $cornerAngle)'; png_path='$i/w${ij}_$comp.$sample.err.png'" plot_relerrors.gpi
            done
            for comp in {0..1}; do
                gnuplot -e "run_dir='$i'; sample_pt=$sample; fluctuation=$ij; component=$comp; png_path='$i/w${ij}_$comp.$sample.png'" plot_displacements.gpi
            done
        done
    done
    for ij in {0..2}; do
        gnuplot -e "run_dir='$i'; fluctuation=$ij" plot_maxstrains.gpi
    done
done

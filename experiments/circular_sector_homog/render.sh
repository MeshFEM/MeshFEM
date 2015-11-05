# render the max strain field for fluctuation w_00
# GMSH's eigenvalue rendering is difficult to use, so we get the max eigenvalue using msh_processor
for msh in results/skip_*/deg_*/homog_20.msh; do
    $MeshFEM/tools/msh_processor $msh -e "strain w_ij 0" --elementAverage --eigenvalues --max -o tmp.msh
    gmsh -n tmp.msh render_scalarfield.opt
    convert -trim render.png ${msh%.msh}.maxstrain.w_00.png

    $MeshFEM/tools/msh_processor $msh -e "strain w_ij 1" --elementAverage --eigenvalues --max -o tmp.msh
    gmsh -n tmp.msh render_scalarfield.opt
    convert -trim render.png ${msh%.msh}.maxstrain.w_11.png

    rm tmp.msh
done

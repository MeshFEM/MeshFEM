gmsh -n sample_poisson_2D.msh render.geo
mv u_perspective.png sample_poisson_2D.u.png
mv u_top.png         sample_poisson_2D.u.top.png

for thick in {0.25,0.50,0.75,1.00,1.50,2.00,2.50,3.00}; do
    name=sample_poisson_${thick}_extrude
    gmsh -n $name.msh render.geo
    mv u_perspective.png $name.u.png
    mv u_top.png         $name.u.top.png
done

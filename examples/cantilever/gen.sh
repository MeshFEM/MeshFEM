#!/bin/zsh
for i in {0..4}; do ../../mesh_convert bar3D_quad.obj -q$i bar3D_tri_$i.poly; done
for i in {0..4}; do tetgen -Y -F -qp -a$((0.125**$i)) bar3D_tri_$i.poly; done
for i in {0..4}; do ../../mesh_convert bar3D_tri_$i.1.node bar_tet_$i.msh; done
rm *.node *.poly *.ele

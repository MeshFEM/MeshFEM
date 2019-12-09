# for thick in {0.06125,0.125,0.25,0.50,1.00}; do
#     name=pattern_extrude_${thick}_thick
#     $MeshFEM/mesh_convert fine_adaptive_meshes/smooth_30.msh -e$thick --quadAspectThreshold 2.0 $name.surf.msh;
#     # $MeshFEM/mesh_convert fine_adaptive_meshes/smooth_30.msh --extrudeTriQuad $thick $name.poly;
#     # # tetgen -Y -F -pqa0.0001 $name.poly;
#     # tetgen -F -pqa0.0001 $name.poly;
#     # $MeshFEM/mesh_convert $name.*.node $name.msh;
#     # rm $name*.{node,ele,poly};
# done

for thick in {0.06125,0.125}; do
    name=pattern_extrude_${thick}_thick_2x2
    $MeshFEM/mesh_convert fine_adaptive_tilings/smooth_30_2x2.msh -e$thick --quadAspectThreshold 2.0 $name.surf.msh;
    $MeshFEM/mesh_convert fine_adaptive_tilings/smooth_30_2x2.msh --extrudeTriQuad $thick $name.poly;
    tetgen -F -pqa0.0001 $name.poly;
    $MeshFEM/mesh_convert $name.*.node $name.msh;
    rm $name*.{node,ele,poly};
done

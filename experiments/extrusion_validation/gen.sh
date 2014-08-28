for thick in {0.25,0.50,0.75,1.00,1.50,2.00,2.50,3.00}; do
    name=sample_poisson_${thick}_extrude
    $MeshFEM/mesh_convert sample_poisson.msh -e$thick $name.poly;
    tetgen -Y -F -pqa0.0001 $name.poly;
    $MeshFEM/mesh_convert $name.*.node $name.msh;
    rm $name*.{node,ele,poly};
    $MeshFEM/Simulate_cli -m ProJet7000_2D.material.material $name.msh  -b compression_relative_3D.bc -o $name.msh;
done

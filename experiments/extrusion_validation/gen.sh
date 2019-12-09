Simulate_cli -m ProJet7000_2D.material.material sample_poisson.msh  -b compression_relative_2D.bc -o sample_poisson_2D.msh;
for thick in {0.25,0.50,0.75,1.00,1.50,2.00,2.50,3.00}; do
    name=sample_poisson_${thick}_extrude
    mesh_convert sample_poisson.msh -e$thick $name.poly;
    tetgen -Y -F -pqa0.0001 $name.poly;
    mesh_convert $name.*.node $name.msh;
    rm $name*.{node,ele,poly};
    Simulate_cli -m ProJet7000_2D.material.material $name.msh  -b compression_relative_3D.bc -o $name.msh;
done

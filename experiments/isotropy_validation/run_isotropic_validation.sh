pattern=$1
~/microstructures/pattern_optimization/Inflator_cli $pattern -c 10 -S1 -o $pattern.S1.msh 
~/MeshFEM/tools/IsotropicValidation -t -n5000 -m ~/MeshFEM/experiments/fit_validation/ProJet7000_2D.material $pattern.S1.msh > $pattern.S1.isovalidate.txt 2>&1
~/microstructures/pattern_optimization/Inflator_cli $pattern -c 10 -S2 -o $pattern.S2.msh 
~/MeshFEM/tools/IsotropicValidation -t -n5000 -m ~/MeshFEM/experiments/fit_validation/ProJet7000_2D.material $pattern.S2.msh > $pattern.S2.isovalidate.txt 2>&1

#!/usr/bin/env zsh
mkdir -p twisted/linear twisted/quadratic sin/linear sin/quadratic
for res in {2..256..2}; do
    ../../Poisson_cli <(../../tools/grid ${res}x${res} -t -m'0,0' -M'1,1' /dev/stdout) -b ../../examples/boundary_conditions/poisson/twisted_square.bc -d1 -o twisted/linear/$res.msh
    ../../Poisson_cli <(../../tools/grid ${res}x${res} -t -m'0,0' -M'1,1' /dev/stdout) -b ../../examples/boundary_conditions/poisson/twisted_square.bc -d2 -o twisted/quadratic/$res.msh
    ../../Poisson_cli <(../../tools/grid ${res}x${res} -t -m'0,0' -M'1,1' /dev/stdout) -b ../../examples/boundary_conditions/poisson/sin_top.bc -d1 -o sin/linear/$res.msh
    ../../Poisson_cli <(../../tools/grid ${res}x${res} -t -m'0,0' -M'1,1' /dev/stdout) -b ../../examples/boundary_conditions/poisson/sin_top.bc -d2 -o sin/quadratic/$res.msh
done

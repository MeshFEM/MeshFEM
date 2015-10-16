#!/usr/bin/env zsh
mkdir -p sin/linear sin/quadratic
for res in {2..256..2}; do
    ../../Simulate_cli <(../../tools/grid ${res}x${res} -t -m'0,0' -M'1,1' /dev/stdout) -m $MICRO_DIR/materials/B9Creator.material -b sin_top.bc -d1 -o sin/linear/$res.msh
    ../../Simulate_cli <(../../tools/grid ${res}x${res} -t -m'0,0' -M'1,1' /dev/stdout) -m $MICRO_DIR/materials/B9Creator.material -b sin_top.bc -d2 -o sin/quadratic/$res.msh
done

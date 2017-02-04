#!/usr/bin/env zsh
for i in {0..4}; do
    ../../tools/grid $((5 * 2**$i))x$((2**$i))x$((2**$i)) -t bar_tet_$i.msh
done

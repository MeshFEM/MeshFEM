#!/bin/bash
# Args:
#   Mesh title
#   plot number
gnuplot <<HERE
set terminal pdf
set output 'plot_$2.pdf'
unset key
set title "$1 Maximum Stress Under Refinement"
set xlabel 'grid size'
set ylabel 'maximum stress'
set yrange [0.015:0.065]
plot 'data.txt' using 5:3 with lines, '< cat data.txt | head -n$2 | tail -n1' using 5:3 pt 7 ps 2
HERE

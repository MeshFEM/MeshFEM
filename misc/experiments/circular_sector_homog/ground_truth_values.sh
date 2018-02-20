# Get the ground-truth values associated with a circular sector convergence
# test run (including the cone angle for convenience)
# The result is the following tab-separated values:
#   mesh_cone_angle Ex Ey nu_yx mu_xy
# The "ground truth" values for the moduli are taken from the highest resolution deg 2 run.
# Usage:
#   ground_truth_values.sh run_dir
# Example:
#   ground_truth_values.sh results/skip_0
sort -g $1/deg_2.txt | tail -n1 | cut -f2,6-

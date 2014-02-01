#!/usr/bin/env python
import argparse
import numpy as np
from mako.template import Template
from mako.runtime import Context
import os.path
from math import pi
from subprocess import check_call
from timethis import timethis
from ColumnRanges import extract_column_ranges

TMP_DIR = "./";
EPS = 1e-3;

@timethis
def generate_R_script(csv_file, prefix, rank, err_bound, Ne, Nt,
        angle_index, ratio_index, laminate_index,
        xmin, xmax, ymin, ymax):
    r_template = Template(filename="plot.mako");
    r_file = os.path.join(TMP_DIR, "plot.r");
    if prefix == "":
        prefix = "p{}_e{}_layer{}".format(rank, err_bound, laminate_index);
    else:
        prefix = "{}_p{}_e{}_layer{}".format(prefix, rank, err_bound,
                laminate_index);

    with open(r_file, 'w') as fout:
        r_template.get_def("header").render_context(Context(
            fout, csv_file = csv_file, err_bound = err_bound));

        subtitle = "";

        if angle_index >= 0:
            angle = pi / Ne * angle_index; 
            subtitle += " angle={} ".format(angle)
            prefix += "_angle{}".format(angle_index);
            r_template.get_def("filter_data").render_context(Context(
                fout, field_name = "Angle_{}".format(laminate_index),
                upper_bound = angle + EPS,
                lower_bound = angle - EPS));

        if ratio_index >= 0:
            ratio = 1.0 / (Ne+1) * (ratio_index + 1);
            subtitle += " ratio={} ".format(ratio)
            prefix += "_ratio{}".format(ratio_index);
            r_template.get_def("filter_data").render_context(Context(
                fout, field_name = "Ratio_{}".format(laminate_index),
                upper_bound = ratio + EPS,
                lower_bound = ratio - EPS));

        r_template.get_def("point_plot").render_context(Context(
            fout, x_col = "Lambda", y_col = "Mu", w_col = "Error",
            title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                    subtitle));

        if xmin < xmax and ymin < ymax:
            r_template.get_def("set_range").render_context(Context(
                fout,
                x_col_min = xmin, x_col_max = xmax,
                y_col_min = ymin, y_col_max = ymax));

        r_template.get_def("save_plot").render_context(Context(
            fout, width=10, height=6,
            out_name = "{}_param.pdf".format(prefix)));

    return r_file;

@timethis
def plot(csv_file, prefix, rank, err_bound, Ne, Nt,
        angle_index, ratio_index, laminate_index,
        xmin, xmax, ymin, ymax):
    r_file = generate_R_script(csv_file, prefix, rank, err_bound, Ne, Nt,
            angle_index, ratio_index, laminate_index, xmin, xmax, ymin, ymax);
    command = "Rscript {}".format(r_file);
    check_call(command.split());

def parse_args():
    parser = argparse.ArgumentParser(description="Plot csv files");
    parser.add_argument("csv_file", help="target csv file");

    # Global setting
    parser.add_argument("--rank", help="number of laminations",
            type=int, required=True);
    parser.add_argument("--prefix", help="prefix of output file",
            default="");
    parser.add_argument("-E", "--error-bound",
            help="only plot data within this error bound", type=float,
            default=0.1);
    parser.add_argument("--Ne", help="number of discrete angles", type=int);
    parser.add_argument("--Nt", help="number of discrete material ratios",
            type=int);

    # Index setting
    parser.add_argument("--angle-index", help="select angle to plot", type=int,
            default=-1);
    parser.add_argument("--ratio-index", help="select material ratio to plot", type=int,
            default=-1);
    parser.add_argument("--laminate-index", help="select laminate to plot", type=int,
            default=0);

    # Range settings
    # By default xmin > xmax and ymin > ymax, so range is not used.
    parser.add_argument("--xmin", help="minimum of x", type=float, default= 1.0);
    parser.add_argument("--xmax", help="maximum of x", type=float, default=-1.0);
    parser.add_argument("--ymin", help="minimum of y", type=float, default= 1.0);
    parser.add_argument("--ymax", help="maximum of y", type=float, default=-1.0);
    args = parser.parse_args();
    return args;

def main():
    args = parse_args();
    plot(args.csv_file, args.prefix, args.rank, args.error_bound,
            args.Ne, args.Nt,
            args.angle_index, args.ratio_index, args.laminate_index,
            args.xmin, args.xmax, args.ymin, args.ymax);
    timethis.summarize();

if __name__ == "__main__":
    main();


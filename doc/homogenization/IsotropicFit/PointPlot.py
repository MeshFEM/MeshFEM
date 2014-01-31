#!/usr/bin/env python
import argparse
import numpy as np
from mako.template import Template
from mako.runtime import Context
import os.path
from subprocess import check_call
from timethis import timethis

TMP_DIR = "./";

@timethis
def generate_R_script(csv_file, prefix, rank, err_bound):
    r_template = Template(filename="plot.mako");
    r_file = os.path.join(TMP_DIR, "plot.r");
    if prefix == "":
        prefix = "p{}_e{}".format(rank, err_bound);
    else:
        prefix = "{}_p{}_e{}".format(prefix, rank, err_bound);

    with open(r_file, 'w') as fout:
        r_template.get_def("header").render_context(Context(
            fout, csv_file = csv_file, err_bound = err_bound));

        # Plot all data points.
        r_template.get_def("point_plot").render_context(Context(
            fout, x_col = "Lambda", y_col = "Mu", w_col = "Error",
            title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                    "Lambda vs Mu",
            out_name = "{}_param.pdf".format(prefix),
            facet_1 = None, facet_2 = None));

        r_template.get_def("histogram").render_context(Context(
            fout, w_col = "Error",
            title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                    "Fitting error",
            out_name = "{}_error.pdf".format(prefix),
            facet_1 = None, facet_2 = None));

        for p in range(rank):
            r_template.get_def("point_plot").render_context(Context(
                fout, x_col = "Lambda", y_col = "Mu", w_col = "Error",
                title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                        "Lambda vs Mu given alpha{} (row) and theta{} (col)".format(p,p),
                out_name = "{}_alpha{}_theta{}_param.pdf".format(prefix, p, p),
                facet_1 = "Angle_{}".format(p),
                facet_2 = "Ratio_{}".format(p)));

            r_template.get_def("histogram").render_context(Context(
                fout, w_col = "Error",
                title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                        "Fitting error given alpha{} (row) and theta{} (col)".format(p,p),
                out_name = "{}_alpha{}_theta{}_error.pdf".format(prefix, p, p),
                facet_1 = "Angle_{}".format(p),
                facet_2 = "Ratio_{}".format(p)));

    return r_file;

@timethis
def plot(csv_file, prefix, rank, err_bound):
    r_file = generate_R_script(csv_file, prefix, rank, err_bound);
    command = "Rscript {}".format(r_file);
    check_call(command.split());

def parse_args():
    parser = argparse.ArgumentParser(description="Plot csv files");
    parser.add_argument("csv_file", help="target csv file");
    parser.add_argument("--rank", help="number of laminations",
            type=int, required=True);
    parser.add_argument("--prefix", help="prefix of output file",
            default="");
    parser.add_argument("-E", "--error-bound",
            help="only plot data within this error bound", type=float,
            default=0.1);
    args = parser.parse_args();
    return args;

def main():
    args = parse_args();
    plot(args.csv_file, args.prefix, args.rank, args.error_bound);
    timethis.summarize();

if __name__ == "__main__":
    main();

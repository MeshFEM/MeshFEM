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
                    "Lambda vs Mu"));
        r_template.get_def("save_plot").render_context(Context(
            fout, width=10, height=6,
            out_name = "{}_param.pdf".format(prefix)));

        # Histogram of residual errors
        r_template.get_def("histogram").render_context(Context(
            fout, w_col = "Error",
            title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                    "Fitting error"));
        r_template.get_def("save_plot").render_context(Context(
            fout, width=10, height=6,
            out_name = "{}_error.pdf".format(prefix)));

        for p in range(rank):
            extended_prefix = "{}_alpha{}_theta{}".format(prefix, p, p);
            # Plat data points with facets
            r_template.get_def("point_plot").render_context(Context(
                fout, x_col = "Lambda", y_col = "Mu", w_col = "Error",
                title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                        "Lambda vs Mu given alpha{} (row) and theta{} (col)".format(p,p)));

            r_template.get_def("add_facet").render_context(Context(
                fout, 
                facet_1 = "Angle_{}".format(p),
                facet_2 = "Ratio_{}".format(p)));

            r_template.get_def("save_plot").render_context(Context(
                fout, width=20, height=12,
                out_name = "{}_param.pdf".format(extended_prefix)));

            # Histogram with facets
            r_template.get_def("histogram").render_context(Context(
                fout, w_col = "Error",
                title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                        "Fitting error given alpha{} (row) and theta{} (col)".format(p,p)));

            r_template.get_def("add_facet").render_context(Context(
                fout, 
                facet_1 = "Angle_{}".format(p),
                facet_2 = "Ratio_{}".format(p)));

            r_template.get_def("save_plot").render_context(Context(
                fout, width=20, height=12,
                out_name = "{}_error.pdf".format(extended_prefix)));

        for pi in range(rank):
            for pj in range(pi+1,rank):
                if pi == pj:
                    continue;
                extended_prefix = "{}_alpha{}_alpha{}".format(prefix, pi, pj);
                r_template.get_def("point_plot").render_context(Context(
                    fout, x_col = "Lambda", y_col = "Mu", w_col = "Error",
                    title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                            "Lambda vs Mu given alpha{} (row) and alpha{} (col)".format(pi,pj)));
                r_template.get_def("add_facet").render_context(Context(
                    fout,
                    facet_1 = "Angle_{}".format(pi),
                    facet_2 = "Angle_{}".format(pj)));
                r_template.get_def("save_plot").render_context(Context(
                    fout, width=20, height=12,
                    out_name = "{}_param.pdf".format(extended_prefix)));

                extended_prefix = "{}_theta{}_theta{}".format(prefix, pi, pj);
                r_template.get_def("point_plot").render_context(Context(
                    fout, x_col = "Lambda", y_col = "Mu", w_col = "Error",
                    title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                            "Lambda vs Mu given theta{} (row) and theta{} (col)".format(pi,pj)));
                r_template.get_def("add_facet").render_context(Context(
                    fout,
                    facet_1 = "Ratio_{}".format(pi),
                    facet_2 = "Ratio_{}".format(pj)));
                r_template.get_def("save_plot").render_context(Context(
                    fout, width=20, height=12,
                    out_name = "{}_param.pdf".format(extended_prefix)));

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

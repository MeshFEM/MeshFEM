#!/usr/bin/env python
import argparse
import numpy as np
from mako.template import Template
from mako.runtime import Context
import os.path
from subprocess import check_call
from timethis import timethis
from ggplot import ggplot

TMP_DIR = "./";

def update_prefix(prefix, rank, err_bound):
    if len(prefix) > 0:
        prefix += "_"
    prefix = prefix + "p{}_e{}".format(rank, err_bound);
    return prefix;

@timethis
def generate_R_script(csv_file, prefix, rank, err_bound, output_format):
    prefix = update_prefix(prefix, rank, err_bound);
    plot = ggplot();
    plot.set_data(csv_file);
    plot.filter_data("Error", 0.0, err_bound);

    base_A_young = plot.get_col("Young_A")[0];
    base_B_young = plot.get_col("Young_B")[0];
    base_A_poisson = plot.get_col("Poisson_A")[0];
    base_B_poisson = plot.get_col("Poisson_B")[0];

    plot.scatter_plot("Lambda", "Mu", "Error");
    plot.title("Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
            "Lambda vs Mu");
    plot.save_plot("{}_lambda_mu.{}".format(prefix, output_format));

    plot.scatter_plot("Youngs_modulus", "Poisson_ratio", "Error");
    plot.title("Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
            "Young's modulus vs Poisson ratio");
    plot.draw_point(base_A_young, base_A_poisson);
    plot.draw_text(base_A_young, base_A_poisson, " A");
    plot.draw_point(base_B_young, base_B_poisson);
    plot.draw_text(base_B_young, base_B_poisson, " B");
    plot.save_plot("{}_young_poisson.{}".format(prefix, output_format));

    plot.plot();

@timethis
def generate_R_script_old(csv_file, prefix, rank, err_bound, output_format):
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
            out_name = "{}_lambda_mu.{}".format(prefix, output_format)));

        r_template.get_def("point_plot").render_context(Context(
            fout, x_col = "Youngs_modulus", y_col = "Poisson_ratio", w_col = "Error",
            title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                    "Young vs Poisson"));
        r_template.get_def("draw_ave_points").render_context(Context(
            fout, x_col = "Young_A", y_col = "Poisson_A", label="A"));
        r_template.get_def("draw_ave_points").render_context(Context(
            fout, x_col = "Young_B", y_col = "Poisson_B", label="B"));
        r_template.get_def("save_plot").render_context(Context(
            fout, width=10, height=6,
            out_name = "{}_young_poisson.{}".format(prefix, output_format)));

        # Histogram of residual errors
        r_template.get_def("histogram").render_context(Context(
            fout, w_col = "Error", num_bins=20,
            title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                    "Fitting error"));
        r_template.get_def("save_plot").render_context(Context(
            fout, width=10, height=6,
            out_name = "{}_error.{}".format(prefix, output_format)));

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
                out_name = "{}_lambda_mu.{}".format(extended_prefix,
                    output_format)));

            r_template.get_def("point_plot").render_context(Context(
                fout, x_col = "Youngs_modulus", y_col = "Poisson_ratio", w_col = "Error",
                title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                        "Young vs Poisson given alpha{} (row) and theta{} (col)".format(p,p)));
            r_template.get_def("draw_ave_points").render_context(Context(
                fout, x_col = "Young_A", y_col = "Poisson_A", label="A"));
            r_template.get_def("draw_ave_points").render_context(Context(
                fout, x_col = "Young_B", y_col = "Poisson_B", label="B"));

            r_template.get_def("add_facet").render_context(Context(
                fout, 
                facet_1 = "Angle_{}".format(p),
                facet_2 = "Ratio_{}".format(p)));

            r_template.get_def("save_plot").render_context(Context(
                fout, width=20, height=12,
                out_name = "{}_young_poisson.{}".format(extended_prefix,
                    output_format)));

            # Histogram with facets
            r_template.get_def("histogram").render_context(Context(
                fout, w_col = "Error", num_bins=20,
                title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                        "Fitting error given alpha{} (row) and theta{} (col)".format(p,p)));

            r_template.get_def("add_facet").render_context(Context(
                fout, 
                facet_1 = "Angle_{}".format(p),
                facet_2 = "Ratio_{}".format(p)));

            r_template.get_def("save_plot").render_context(Context(
                fout, width=20, height=12,
                out_name = "{}_error.{}".format(extended_prefix, output_format)));

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
                    out_name = "{}_lambda_mu.{}".format(extended_prefix,
                        output_format)));

                extended_prefix = "{}_alpha{}_alpha{}".format(prefix, pi, pj);
                r_template.get_def("point_plot").render_context(Context(
                    fout, x_col = "Youngs_modulus", y_col = "Poisson_ratio", w_col = "Error",
                    title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                            "Young vs Poisson given alpha{} (row) and alpha{} (col)".format(pi,pj)));
                r_template.get_def("draw_ave_points").render_context(Context(
                    fout, x_col = "Young_A", y_col = "Poisson_A", label="A"));
                r_template.get_def("draw_ave_points").render_context(Context(
                    fout, x_col = "Young_B", y_col = "Poisson_B", label="B"));
                r_template.get_def("add_facet").render_context(Context(
                    fout,
                    facet_1 = "Angle_{}".format(pi),
                    facet_2 = "Angle_{}".format(pj)));
                r_template.get_def("save_plot").render_context(Context(
                    fout, width=20, height=12,
                    out_name = "{}_young_poisson.{}".format(extended_prefix,
                        output_format)));

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
                    out_name = "{}_lambda_mu.{}".format(extended_prefix,
                        output_format)));

                extended_prefix = "{}_theta{}_theta{}".format(prefix, pi, pj);
                r_template.get_def("point_plot").render_context(Context(
                    fout, x_col = "Youngs_modulus", y_col = "Poisson_ratio", w_col = "Error",
                    title = "Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
                            "Young vs Poisson given theta{} (row) and theta{} (col)".format(pi,pj)));
                r_template.get_def("draw_ave_points").render_context(Context(
                    fout, x_col = "Young_A", y_col = "Poisson_A", label="A"));
                r_template.get_def("draw_ave_points").render_context(Context(
                    fout, x_col = "Young_B", y_col = "Poisson_B", label="B"));
                r_template.get_def("add_facet").render_context(Context(
                    fout,
                    facet_1 = "Ratio_{}".format(pi),
                    facet_2 = "Ratio_{}".format(pj)));
                r_template.get_def("save_plot").render_context(Context(
                    fout, width=20, height=12,
                    out_name = "{}_young_poisson.{}".format(extended_prefix,
                        output_format)));

    return r_file;

@timethis
def plot_old(csv_file, prefix, rank, err_bound, output_format):
    r_file = generate_R_script(csv_file, prefix, rank, err_bound, output_format);
    command = "Rscript {}".format(r_file);
    check_call(command.split());

@timethis
def plot(csv_file, prefix, rank, err_bound, output_format):
    prefix = update_prefix(prefix, rank, err_bound);
    plot = ggplot();
    plot.set_data(csv_file);
    plot.filter_data("Error", 0.0, err_bound);
    #plot.discretize_column("Angle_0");
    #plot.discretize_column("Ratio_0");

    base_A_young = float(plot.get_col("Young_A")[0]);
    base_B_young = float(plot.get_col("Young_B")[0]);
    base_A_poisson = float(plot.get_col("Poisson_A")[0]);
    base_B_poisson = float(plot.get_col("Poisson_B")[0]);

    #plot.scatter_plot("Lambda", "Mu", "Error");
    #plot.title("Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
    #        "Lambda vs Mu");
    #plot.save_plot("{}_lambda_mu.{}".format(prefix, output_format));

    #plot.scatter_plot("Youngs_modulus", "Poisson_ratio", "Error", False);
    #plot.title("Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
    #        "Young's modulus vs Poisson ratio");
    ##plot.set_range(base_A_young, base_B_young, 0.288,base_A_poisson);
    ##plot.set_range(base_A_young, 5, 0.288,base_A_poisson);
    #plot.draw_point(base_A_young, base_A_poisson);
    #plot.draw_text(base_A_young, base_A_poisson, " A");
    #plot.draw_point(base_B_young, base_B_poisson);
    #plot.draw_text(base_B_young, base_B_poisson, " B");
    ##plot.save_plot("{}_young_poisson_error.{}".format(prefix, output_format,
    ##    width=5, height=3));
    #plot.save_plot("{}_young_poisson_error_{}.{}".format(prefix, base_B_young, output_format));

    #ratio_sum = "+".join(["Ratio_{}".format(i) for i in range(rank)]);
    #plot.scatter_plot("Youngs_modulus", "Poisson_ratio", ratio_sum, False);
    #plot.title("Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
    #        "Young's modulus vs Poisson ratio");
    #plot.set_range(base_A_young, base_B_young, 0.288,base_A_poisson);
    #plot.draw_point(base_A_young, base_A_poisson);
    #plot.draw_text(base_A_young, base_A_poisson, " A");
    #plot.draw_point(base_B_young, base_B_poisson);
    #plot.draw_text(base_B_young, base_B_poisson, " B");
    #plot.save_plot("{}_young_poisson_ratio.{}".format(prefix, output_format));

    plot.add_column("Angle_Diff",\
            "pmin(abs(Angle_0 - Angle_1), pi - abs(Angle_0 - Angle_1))");
    plot.scatter_plot("Youngs_modulus", "Poisson_ratio", "Angle_Diff", False);
    plot.title("Rank-{} Laminates (Error < {})\n".format(rank, err_bound) +\
            "Young's modulus vs Poisson ratio");
    plot.draw_point(base_A_young, base_A_poisson);
    plot.draw_text(base_A_young, base_A_poisson, " A");
    plot.draw_point(base_B_young, base_B_poisson);
    plot.draw_text(base_B_young, base_B_poisson, " B");
    plot.save_plot("{}_young_poisson_angle.{}".format(prefix, output_format));

    plot.plot();

def parse_args():
    parser = argparse.ArgumentParser(description="Plot csv files");
    parser.add_argument("csv_file", help="target csv file");
    parser.add_argument("--rank", help="number of laminations",
            type=int, required=True);
    parser.add_argument("--prefix", help="prefix of output file",
            default="");
    parser.add_argument("-E", "--error-bound",
            help="only plot data within this error bound", type=float,
            default=1.0);
    parser.add_argument("--format", default="png");
    args = parser.parse_args();
    return args;

def main():
    args = parse_args();
    plot(args.csv_file, args.prefix, args.rank, args.error_bound, args.format);
    timethis.summarize();

if __name__ == "__main__":
    main();

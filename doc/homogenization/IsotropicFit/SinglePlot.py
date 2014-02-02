#!/usr/bin/env python
import argparse
import json
import numpy as np
from mako.template import Template
from mako.runtime import Context
import os.path
from math import pi
from subprocess import check_call
from timethis import timethis
from ColumnRanges import extract_column_ranges

TMP_DIR = "/tmp/";
EPS = 1e-3;

def load_config(config_file):
    with open(config_file, 'r') as fin:
        config = json.load(fin);
    return config;

def update_prefix(prefix, rank, err_bound, Ne, Nt,
        angle_index, ratio_index, laminate_index):
    if len(prefix) > 0:
        prefix += "_"
    prefix = prefix + "p{}_{}x{}_e{}_layer{}".format(
            rank, Ne, Nt, err_bound, laminate_index);
    if angle_index >= 0:
        prefix += "_alpha{}".format(angle_index);
    if ratio_index >= 0:
        prefix += "_theta{}".format(ratio_index);
    return prefix;

@timethis
def generate_R_script_from_config(config_file):
    config = load_config(config_file);
    def get_config(name, default=None):
        if name in config:
            return config[name];
        else:
            return default;

    prefix    = get_config("prefix");
    rank      = get_config("rank");
    err_bound = get_config("error_bound");
    csv_file  = get_config("csv_file");

    Ne = get_config("num_angles");
    Nt = get_config("num_ratios");

    angle_index    = get_config("angle_index", -1);
    ratio_index    = get_config("ratio_index", -1);
    laminate_index = get_config("laminate_index");

    xmin = get_config("xmin", 1);
    xmax = get_config("xmax",-1);
    ymin = get_config("ymin", 1);
    ymax = get_config("ymax",-1);

    out_dir = get_config("out_dir", "./");

    return generate_R_script(csv_file, prefix, rank, err_bound, Ne, Nt,
            angle_index, ratio_index, laminate_index,
            xmin, xmax, ymin, ymax, out_dir);

@timethis
def generate_R_script(csv_file, prefix, rank, err_bound, Ne, Nt,
        angle_index, ratio_index, laminate_index,
        xmin, xmax, ymin, ymax, out_dir):
    r_template = Template(filename="plot.mako");
    prefix = update_prefix(prefix, rank, err_bound, Ne, Nt,
            angle_index, ratio_index, laminate_index);
    r_file = os.path.join(TMP_DIR, prefix + "_param.r");
    out_name = os.path.join(TMP_DIR, prefix + "_param.pdf");

    with open(r_file, 'w') as fout:
        r_template.get_def("header").render_context(Context(
            fout, csv_file = csv_file, err_bound = err_bound));

        subtitle = "";

        if angle_index >= 0:
            angle = pi / Ne * angle_index; 
            subtitle += " angle={} ".format(angle)
            r_template.get_def("filter_data").render_context(Context(
                fout, field_name = "Angle_{}".format(laminate_index),
                upper_bound = angle + EPS,
                lower_bound = angle - EPS));

        if ratio_index >= 0:
            ratio = 1.0 / (Ne+1) * (ratio_index + 1);
            subtitle += " ratio={} ".format(ratio)
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
            out_name = out_name));

    return r_file, out_name, out_dir;


@timethis
def plot(config_file):
    r_file, out_name, out_dir = generate_R_script_from_config(config_file);
    command = "Rscript {}".format(r_file);
    check_call(command.split());
    assert(os.path.exists(out_name));
    command = "mv {} {}/.".format(out_name, out_dir)
    check_call(command.split());


def parse_args():
    parser = argparse.ArgumentParser(description="Plot csv files");
    parser.add_argument("config_file", help="Plot configuration file");
    args = parser.parse_args();
    return args;

def main():
    args = parse_args();
    plot(args.config_file);
    timethis.summarize();

if __name__ == "__main__":
    main();


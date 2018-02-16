#!/usr/bin/env python
import argparse
import json
import os
import os.path

from ColumnRanges import extract_column_ranges

def get_ranges(directory):
    """
    Extract the range of the first 3 columns of all .csv files in this dir.
    """
    csv_files = [];
    files = os.listdir(directory);
    for f in files:
        name,ext = os.path.splitext(f);
        if ext == ".csv":
            csv_files.append(os.path.join(directory, f));

    min_val, max_val = extract_column_ranges(csv_files);
    return min_val, max_val;

def generate_config_files(directory, err_bound, rank, Ne, Nt, csv_file,
        config_output_dir, plot_output_dir):
    config_files = [];

    csv_file = os.path.abspath(csv_file);
    config_otuput_dir = os.path.abspath(config_output_dir);
    plot_output_dir = os.path.abspath(plot_output_dir);

    min_val, max_val = get_ranges(directory);
    for rank_i in range(rank):
        for angle_i in range(Ne):
            for ratio_i in range(Nt):
                config = {
                        "prefix":"",
                        "rank":rank,
                        "error_bound":err_bound,
                        "csv_file": csv_file,
                        "num_angles": Ne,
                        "num_ratios": Nt,
                        "angle_index": angle_i,
                        "ratio_index": ratio_i,
                        "laminate_index": rank_i,
                        "xmin": min_val[0],
                        "xmax": max_val[0],
                        "ymin": min_val[1],
                        "ymax": max_val[1],
                        "out_dir": plot_output_dir
                        };
                config_file = "p{}_{}x{}_e{}_layer{}_alpha{}_theta{}.config"\
                        .format(rank, Ne, Nt, err_bound, rank_i, angle_i, ratio_i);
                config_file = os.path.join(config_output_dir, config_file);
                with open(config_file, 'w') as fout:
                    json.dump(config, fout, indent=4);
                config_files.append(config_file);
    return config_files;


def parse_args():
    parser = argparse.ArgumentParser(description=\
            "Generate sequence of command using SinglePlot to examine facets.");
    parser.add_argument("--directory", help="directory containing csv files",
            required=True);
    parser.add_argument("-E", "--error-bound", type=float, required=True);
    parser.add_argument("--Ne", help="number of discrete angles", type=int,
            required=True);
    parser.add_argument("--Nt", help="number of discrete material ratios",
            type=int, required=True);
    parser.add_argument("--rank", help="number of laminations",
            type=int, required=True);
    parser.add_argument("--config-outdir", help="output directory for config files")
    parser.add_argument("--plot-outdir", help="output directory for plots");
    parser.add_argument("csv_file", help="target csv file");
    args = parser.parse_args();
    return args;

def main():
    args = parse_args();
    config_files = generate_config_files(
            args.directory, args.error_bound,
            args.rank, args.Ne, args.Nt, args.csv_file,
            args.config_outdir, args.plot_outdir);

    basename, ext = os.path.splitext(args.csv_file);
    path, name = os.path.split(basename);
    job_file = os.path.join(args.config_outdir, name + ".job");
    with open(job_file, 'w') as fout:
        for config_file in config_files:
            fout.write("{}\n".format(config_file));

if __name__ == "__main__":
    main();


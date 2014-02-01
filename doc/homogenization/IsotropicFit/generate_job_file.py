#!/usr/bin/env python
import argparse
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

def generate_job_commands(directory, err_bound, rank, Ne, Nt, csv_file):
    min_val, max_val = get_ranges(directory);
    base_command = "./SinglePlot.py --rank {} -E {} --Ne {} --Nt {}".format(
            rank, err_bound, Ne, Nt);
    index_setting = "--angle-index {} --ratio-index {} --laminate-index {}";
    range_setting = "--xmin={} --xmax={} --ymin={} --ymax={}".format(
            min_val[0], max_val[1], min_val[1], max_val[1]);

    commands = [];
    for rank_i in range(rank):
        for angle_i in range(Ne):
            for ratio_i in range(Nt):
                command = "{} {} {} {}".format(
                        base_command,
                        index_setting.format(angle_i, ratio_i, rank_i),
                        range_setting,
                        csv_file);
                commands.append(command);
    return commands;

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
    parser.add_argument("csv_file", help="target csv file");
    args = parser.parse_args();
    return args;

def main():
    args = parse_args();
    commands = generate_job_commands(args.directory, args.error_bound,
            args.rank, args.Ne, args.Nt, args.csv_file);
    for cmd in commands:
        print(cmd);

if __name__ == "__main__":
    main();


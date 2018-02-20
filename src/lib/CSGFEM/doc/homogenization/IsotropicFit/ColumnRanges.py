#!/usr/bin/env python
import argparse
import csv
import os
import numpy as np
from timethis import timethis

@timethis
def extract_column_ranges(csv_files):
    rows = [];
    for filename in csv_files:
        with open(filename, 'r') as fin:
            num_rows = 0;
            csv_reader = csv.reader(fin);
            for row in csv_reader:
                if num_rows != 0:
                    rows.append(row[:3]);
                num_rows += 1;
    rows = np.array(rows, dtype=float);
    return np.amin(rows, axis=0), np.amax(rows, axis=0);


def parse_args():
    parser = argparse.ArgumentParser(
            description="Extract column ranges from csv files");
    parser.add_argument("csv_files", nargs="+");
    args = parser.parse_args();
    return args;


def main():
    args = parse_args();
    [col_min, col_max] = extract_column_ranges(args.csv_files);
    print(col_min);
    print(col_max);
    timethis.summarize();

if __name__ == "__main__":
    main();

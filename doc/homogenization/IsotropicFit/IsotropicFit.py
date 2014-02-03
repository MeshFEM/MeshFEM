#!/usr/bin/env python

import argparse
import numpy as np
from numpy.linalg import lstsq, norm

from Material import Material
from Material2D import Material2D
from timethis import timethis

@timethis
def process_tensors(filename):
    material = Material.create(2);
    material.load(filename);
    material.fit_isotropic();

    Lambda = material.lame_lambda;
    Mu = material.lame_mu;
    Young = material.youngs_modulus;
    Poisson = material.poisson_ratio;
    Bulk = material.bulk_modulus;
    err = material.error;
    angles = material.angles;
    ratios = material.ratios;

    num_laminates = angles.shape[1];
    fields = np.hstack((Lambda, Mu, Young, Poisson, Bulk, err, angles, ratios));
    field_names = ["Lambda", "Mu", "Youngs_modulus", "Poisson_ratio",
            "Bulk_modulus", "Error"] +\
            ["Angle_{}".format(i) for i in range(num_laminates)] +\
            ["Ratio_{}".format(i) for i in range(num_laminates)];
    assert(len(field_names) == fields.shape[1]);

    return field_names, fields;

@timethis
def dump_csv(csv_file, column_names, data):
    num_rows = data.shape[0];
    num_cols = data.shape[1];
    template = ",".join(["{}" for i in range(num_cols)]) + "\n";
    with open(csv_file, 'w') as fout:
        fout.write(template.format(*column_names));
        for i in range(num_rows):
            fout.write(template.format(*data[i]));

def parse_args():
    parser = argparse.ArgumentParser(description =\
            "Check whether material tensor is isotropic");
    parser.add_argument("tensors", help="Specifies the tensor file",
            default=None);
    parser.add_argument("-o", "--output", help="Output csv file", required=True);
    args = parser.parse_args();
    return args;

def main():
    args = parse_args();
    if args.tensors is not None:
        field_names, fields = process_tensors(args.tensors);

    Err = fields[:,2];
    print("{} tensors processed.".format(len(Err)));
    print("Ave error = {}".format(np.average(Err)));
    print("Max error = {}".format(np.max(Err)));
    print("Min error = {}".format(np.min(Err)));

    assert(field_names[5] == "Error");
    fields = fields[fields[:,2].argsort()[::-1]];

    dump_csv(args.output, field_names, fields);

    timethis.summarize();

if __name__ == "__main__":
    main();

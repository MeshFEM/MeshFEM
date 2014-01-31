#!/usr/bin/env python

import argparse
import numpy as np
from numpy.linalg import lstsq, norm

from MaterialTensor import random_tensor
from MaterialTensor import random_isotropic_tensor
from MaterialTensor import load_tensor
from timethis import timethis

@timethis
def vectorize_tensors(tensors):
    shape = tensors.shape;
    return tensors.reshape((shape[0]*shape[1], shape[2]), order="F");

@timethis
def get_isotropic_parameter_matrix(tensor_size):
    if tensor_size == 3:
        return np.array([
            [1, 2],
            [1, 0],
            [0, 0],
            [1, 0],
            [1, 2],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 2] ], dtype=float);
    else:
        raise NotImplementedError("Only 2D is supported for now.");

@timethis
def compute_tensor_norms(tensors):
    fro_norm = norm(tensors, ord='fro', axis=(0,1));
    return fro_norm;

@timethis
def compute_isotropic_tensor_norms(dim, params):
    Lambda = params[0,:];
    Mu = params[1,:];
    norms = np.sqrt(np.square(Mu * 2 + Lambda) * 2 +\
            np.square(Mu * 2) + np.square(Lambda) * 2);
    return norms;

@timethis
def fit_isotropic_material_tensors(tensors):
    shape = tensors.shape;
    coeff_mat = get_isotropic_parameter_matrix(shape[0]);
    rhs = vectorize_tensors(tensors);
    x, err, rank, s_val = lstsq(coeff_mat, rhs);
    input_norms = compute_tensor_norms(tensors);
    output_norms = compute_isotropic_tensor_norms(2, x);
    err = np.divide(np.sqrt(err), output_norms);
    return x, err;

@timethis
def process_tensors(filename):
    A_stars, angles, ratios = load_tensor(filename);
    num_tensors = A_stars.shape[2];
    params, err = fit_isotropic_material_tensors(A_stars);
    err = err.reshape((1, -1));
    num_laminates = angles.shape[0];

    fields = np.hstack((params.T, err.T, angles.T, ratios.T));
    field_names = ["Lambda", "Mu", "Error"] +\
            ["Angle_{}".format(i) for i in range(num_laminates)] +\
            ["Ratio_{}".format(i) for i in range(num_laminates)];
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

    assert(field_names[2] == "Error");
    fields = fields[fields[:,2].argsort()[::-1]];

    dump_csv(args.output, field_names, fields);

    timethis.summarize();

if __name__ == "__main__":
    main();

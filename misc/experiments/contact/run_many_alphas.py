#!/usr/bin/env python
import subprocess

alpha_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
for alpha in alpha_values:
    print("Running simulation with penalization constant alpha equal to " + str(alpha))

    cmd = ['../../../cmake-build-release/MeshFEM/SimulateWithContact_cli', 'tpipe.msh', '-o', "out_alpha_{:.2e}.msh".format(alpha), '-m', '../../../../../microstructures-github/materials/B9Creator.material',
           '-b', 'boundary_conditions.json', '-a', str(alpha)]
    subprocess.call(cmd)
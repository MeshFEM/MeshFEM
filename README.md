MeshFEM
=======

[![](https://github.com/MeshFEM/MeshFEM/workflows/Build/badge.svg?event=push)](https://github.com/MeshFEM/MeshFEM/actions?query=workflow%3ABuild+branch%3Amaster+event%3Apush)

<img src='http://julianpanetta.com/MeshFEMIcon.png' width='229px' height='220px' />

MeshFEM is a C++ finite element library supporting linear and quadratic
triangle/tetrahedral elements. MeshFEM aims to make it easy to write generic
but efficient code supporting multiple problem dimensions (2D, 3D), basis
functions (linear and quadratic), and number types (e.g.,
higher/lower precision, or automatic differentiation types).
The somewhat redundant name "MeshFEM" was given to distinguish it from an
earlier mesh-free "CSGFEM" codebase I developed.

MeshFEM includes efficient simplicial mesh data structures for representing and
traversing triangle/tetrahedral meshes and their boundaries, code for building
and efficiently manipulating sparse matrices, efficient quadrature routines,
and support for fourth-order tensors and tensor fields.

It also provides the linear elasticity solver, periodic homogenization routines
for linear elasticity, and material field optimization implementation that we used in our
[Elastic Textures paper](http://julianpanetta.com/publication/elastic_textures/).

Preliminary [Python bindings](#user-content-python-bindings) are implemented along with a
mesh viewer and vector/scalar field visualization library for Jupyter
Notebook/JupyterLab.

Dependencies
------------
Dependencies *not* included (must be installed separately):

- CHOLMOD/UMFPACK (SuiteSparse)

Dependencies included directly as external projects:

- [Boost](https://github.com/Orphis/boost-cmake) ** Note: installing this separately is recommended to avoid a large download **
- [json](https://github.com/nlohmann/json)
- [triangle](https://www.cs.cmu.edu/~quake/triangle.html)
- [tinyexpr](https://github.com/codeplea/tinyexpr)
- [Eigen](https://github.com/eigenteam/eigen-git-mirror)
- [TBB](https://github.com/01org/tbb)
- [pybind11](https://github.com/pybind/pybind11)

Optional dependencies (without these, certain parts will be omitted from the build):

- [Ceres Solver](http://ceres-solver.org)

Dependencies for running the Jupyter notebooks: [see here](#running-the-jupyter-notebooks).

<a name='building'></a>
Obtaining and Building
----------------------
After installing the non-bundled dependencies, use the following commands on
Mac or Linux to download and build MeshFEM:

```
git clone --recursive https://github.com/jpanetta/MeshFEM
cd MeshFEM
mkdir build && cd build
cmake ..
make -j(# of jobs)
```

I would recommend instead using the [Ninja build system](https://ninja-build.org), which
means changing the last two lines to `cmake .. -GNinja` and `ninja`.

Running the Jupyter Notebooks
-----------------------------
Preliminary Python bindings for parts of the MeshFEM codebase have been
generated using [pybind11](https://github.com/pybind/pybind11). They should already have been built
and installed in the `python` directory when you [built the main project](#user-content-building).

We include some example Jupyter notebooks to demonstrate some of the bound
functionality. Please follow the instructions below to get these notebooks and
the visualization code running.

### JuptyterLab and Extensions

To run the Jupyter notebooks, you will need to install JupyterLab and
[my fork](https://github.com/jpanetta/pythreejs) of the `pythreejs` library.
JupyterLab can be installed through `pip`, and the following commands should
work on both macOS and Ubuntu:

```bash
pip3 install wheel # Needed if installing in a virtual environment
pip3 install jupyterlab==1.2.6 traitlets==4.3.3
# If necessary, follow the instructions in the warnings to add the Python user
# bin directory (containing the 'jupyter' binary) to your PATH...
jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1.0

git clone https://github.com/jpanetta/pythreejs
cd pythreejs
pip3 install -e .
jupyter labextension link ./js

pip3 install matplotlib scipy
```

Launch Jupyter lab from the root python directory:
```bash
cd python
jupyter lab
```

<a name="python-bindings"></a>
Example Usage (Python Bindings)
-------------------------------
In the future, I hope to expose all MeshFEM functionality with Python wrappers.
For now, only a few high-level portions of the code have been bound.
To see what is possible with the current bindings,
please check out the included notebooks
[`python/examples/Homogenization.ipynb`](https://github.com/jpanetta/MeshFEM/blob/master/python/examples/Homogenization.ipynb), and
[`python/examples/GeodesicsInHeat.ipynb`](https://github.com/jpanetta/MeshFEM/blob/master/python/examples/GeodesicsInHeat.ipynb).

Example Usage (C++ Binaries)
----------------------------
The traditional way to interact with the library is by implementing and running C++ binaries.
Several pre-made C++ binaries have been provided to solve linear elasticity problems and run
periodic homogenization.

Linear elasticity simulation
----------------------------
The `Simulate_cli` binary is used to solve linear elasticity problems. To use it, specify an input mesh,
its material properties, and boundary conditions (forces, clamps) to apply.

### Material specification files

#### Isotropic material

Example with a file named **`B9Creator.material`**:

```json
{
    "type": "isotropic_material",
    "dim": 3,
    "young": 200.0,
    "poisson": 0.35
}
```

Orthotropic and general anisotropic materials can also be specified using a `.material` file.

#### Spatially varying material fields
Spatially varying isotropic and (axis-aligned) orthotropic materials can be specified by
passing a `.msh` file instead of a `.material`. This `.msh` file should have identical geometry (vertices and elements) as
the simulation and should also contain per-element scalar fields describing the elastic moduli. For an isotropic material field,
the names of these scalar fields must be `E` and `nu`, optionally prefixed by the string passed as the `matFieldName` command line argument.
For a 3D orthotropic material, the scalar fields should be called  `E_x`, `E_y`, `E_z`, `nu_yx`, `nu_zx`, `nu_zy`, `mu_yz`, `mu_zx`, and `mu_xy`;
for a 2D orthotropic material, they should be called  `E_x`, `E_y`, `nu_yx`, and `mu`.

### Boundary condition files

```json
{
    "no_rigid_motion": false,
    "regions": [
            {   "type": "dirichlet",
                "value": [ 0, 0, 0],
                "box%": { "minCorner": [-0.0001, -0.0001, -0.0001], "maxCorner": [0.0001, 1.0001, 1.0001] }
            },
            {   "type": "force",
                "value": [0, -10, 0],
                "box%": { "minCorner": [ 0.9999, -0.0001, -0.0001], "maxCorner": [1.0001, 1.0001, 1.0001] }
            }
    ]
}
```

Possible types (non-exhaustive list):

- `dirichlet`: target displacement.
- `force`: directional load (total).
- `traction`: directional load (per-unit).
- `presssure`: load along the normal direction.

**Tip**: Use `dirichletxy` to fix only the X and Y component of a region (then value[0:2] will be used). The same component specification syntax can be used with all boundary condition types.

Rectangular box region specification:

- `box`: the corners are specified in absolute coordinates.
- `box%`: the corners are specified relative to the bounding box of the input mesh (which would correspond to `{ "minCorner": [0, 0, 0], "maxCorner": [1, 1, 1] }`).

**Units**:
- `mm` for node positions
- `N` for forces
- `MPa` for Young's modulus and traction (same as `N/mm^2`)

You can also set Dirichlet boundary conditions per boundary elements, by specifying the vertices belonging to each boundary element (edge in 2D, triangle in 3D), in arbitrary order. Example:

```json
{
    "no_rigid_motion": false,
    "regions": [
        {
            "element vertices": [
                [ 0, 3 ],
                [ 2, 3 ],
                [ 2, 4 ],
                [ 4, 5 ],
                [ 5, 6 ],
                [ 6, 7 ],
                [ 0, 8 ],
                [ 7, 8 ]
            ],
            "type": "dirichlet elements",
            "value": [
                "cos(y)",
                "sin(x)",
                "0"
            ]
        }
    ]
}
```

The `no_rigid_motion` option enables/disables constraints on the global
rigid translation and rotation described as described in our
[Worst-Case Structural Analysis paper](http://julianpanetta.com/publication/worst_case_analysis/).
Note that these constraints change the static equilibrium system from positive semidefinite to an
indefinite KKT system, which is significantly more expensive to solve. Whenever
possible, it is recommended to pin down the rigid motion by other means (using
carefully crafted Dirichlet conditions).

### Running the simulation

    ./Simulate_cli -m B9Creator.material -b loads.bc -o output.msh <input_mesh>

The only permitted output file format is `.msh`. This file includes several
vector and tensor fields describing the solution, and can be viewed in
[Gmsh](http://gmsh.info).

Accepted input file formats: (non-exhaustive list):

- Tetrahedral meshes:
  - `.msh`, with tets only (no triangles).
  - `.mesh`, medit file format.
- Triangle meshes:
  - `.off`
  - `.obj`
  - `.stl`

Output fields:

- `u`: per-vertex displacement.
- `load`: per-vertex external forces.
- `Ku`: per-vertex actual force applied to the shape (including the forces applied to enforce no-rigid-motion and Dirichlet constraints).
- `strain`: per-element strain tensor.
- `stress`: per-element stress tensor.

**Note**: in the Gmsh format, vector and tensor fields are always stored
as 3D quantities, even for 2D problems (in which case they are padded with 0).

### Post-processing

To interpret the results of a simulation, either load it in [Gmsh](http://gmsh.info) or
use the `tools/msh_processor`. E.g.:

    $MeshFEM/build/tools/msh_processor in.msh -e ‘stress’ --eigenvalues --max --max

Will compute the greatest maximum principal stress in the structure.

    $MeshFEM/build/tools/msh_processor in.msh -e ‘stress’ --eigenvalues --max -o out.msh

Will write a file `out.msh` with a scalar field containing the maximum principal stress of each element.


Homogenization
--------------
Given a base material, you can run periodic homogenization on a periodic microstructure (2D or 3D) in a
unit cell (square in 2D, cube in 3D) using the `PeriodicHomogenization_cli` binary.

Using the `DeformedCells_cli` binary, this unit cell can be distorted by a
linear warping transformation prior to homogenizing (to homogenize
parallelogram/parallelepiped tilings):

    DeformedCells_cli examples/meshes/square_hole.off -m examples/materials/B9Creator.material --homogenize --jacobian '2 0 0 1' --transformVersion

Explanation of the arguments:

- ` --jacobian 'xx xy yx yy'`. This is the Jacobian of the deformation that maps the unit square/cube to the distorted configuration.

- `--transformVersion`.
By default, `DeformedCells_cli` actually warps the periodic mesh into a
parallelogram and then runs homogenization. If you pass `--transformVersion`,
it instead solves the transformed homogenization problem over the original
undeformed unit cell (by transforming the base material's elasticity tensor
accordingly and transforming the resulting homogenized tensor back). Both
approaches give identical results up to roundoff error.

- It has an additional mode for the 2D case (`--parametrizedTransform`) where a
  sequence of deformations are read from stdin (one per line) and the
  resulting tensor is output for each. These deformations are parametrized by
  `theta lambda`, which specifies the Jacobian $$J = Rot(\theta) [\lambda\;
  0; 0\; 1] Rot(\theta)^T$$.

Notes for Windows (Thanks to [Christopher Brandt](https://people.epfl.ch/christopher.brandt))
---------------------------------------------
Getting MeshFEM to run in Visual Studio 2019:

1) This is specifically for VS2019; older versions would need significant changes in the code
2) Prerequisites:
   - Download boost, extract it, and use the following to commands in the boost root folder in a terminal:
       ```sh
       bootsrap
       bjam --toolset=msvc-14.2 --build-type=complete --prefix=[installation dir]
       ```
     More details on the installation can be [found here](https://theboostcpplibraries.com/introduction-installation).
     Add the folder `[installation dir]` as an environment variable called `BOOST_ROOT`,
     add the folder `[installation dir]\lib` as an environment variable called `BOOST_LIBRARYDIR`
   - Download or clone the CMake version of suitesparse [from here](https://github.com/jlblancoc/suitesparse-metis-for-windows) and then:
       - Configure and Generate the project files for Visual Studio 16 2019 - x64. There are some options
         (like using CUDA and OpenMP), but these depend on your preferences.
          Just make sure you set the installation path (`CMAKE_INSTALL_PREFIX`) to something you remember
       - Open the generated project file, build `ALL_BUILD` and then `INSTALL`.
         The libraries, shared libraries, and include files will now be found in the installation path.
         To later be able to run executables without copying the `.dll`s into the program folder, you can
         add the subdirectories `/bin` and `/lib64/lapack_blas_windows` to your `PATH` environment variable.
         IMPORTANT: after building and installing you have to rename (or duplicate with different names)
         the following `.lib` libraries in the `/lib` subdirectory:
         ```
             libcholmod.lib -> cholmod.lib
             libccolamd.lib -> ccolamd.lib
             libcolamd.lib -> colamd.lib
             libamd.lib -> amd.lib
             libcamd.lib -> camd.lib
             libumfpack.lib -> umfpack.lib
         ```
       - Now add the directory `[suitesparse installation dir]/include/suitesparse` as an environment
         variable called `SUITESPARSE_INC`, `[suitesparse installation dir]/lib` as `SUITESPARSE_LIB`
         and `[suitesparse installation dir]` as `SUITESPARSE_ROOT`.
3) Configure and generate the project files for MeshFEM
   - Aside from a warning related to the `BOOST_ROOT` environment variable, this worked
   without any additional changes (of course only after carefully following the above instructions).
4) Open the `MeshFEM.sln` and try to compile. You will get quite a lot of errors:
   - For some reason I had to manually add the include directory of boost for the MeshFEM project
       (these are found in the boost `[installation dir]\include\boost_XX_X\`)
   - There is a problem when compiling `tinyexpr` in Visual Studio ("initializer not constant")
   with a very simple workaround [described here](https://github.com/codeplea/tinyexpr/issues/34):
       - add these two lines anywhere above `static const te_variable functions[] = {}`:
           ```
           #pragma function (ceil)
           #pragma function (floor)
           ```

Acknowledgements
----------------
Thanks to [Jeremie Dumas](https://www.jdumas.org) for reorganizing the codebase
and transitioning it to a CMake build system.

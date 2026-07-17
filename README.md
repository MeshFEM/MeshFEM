<img src='https://julianpanetta.com/meshfem_logo.jpg' width='500px' />

MeshFEM is a C++ nonlinear finite element library supporting linear and quadratic
triangle/tetrahedral elements. MeshFEM aims to make it easy to write generic
but efficient code supporting multiple problem dimensions (2D, 3D), basis
functions (linear and quadratic), element types (shell, solid, surface parametrization, hinge), and number types (e.g.,
higher/lower precision, or automatic differentiation types).
The somewhat redundant name "MeshFEM" was given to distinguish it from an
earlier mesh-free "CSGFEM" codebase from which it evolved.

This repository has been overhauled to provide the features described
in our [SIGGRAPH 2026 paper](https://doi.org/10.1145/3811386):
> Haleh Mohammadian, Xinzhuo Hu, Roi Poranne, and Julian Panetta.
> **MeshFEM: A Block-accelerated Solver for Nonlinear Finite Elements.**
> *ACM Transactions on Graphics (Proceedings of SIGGRAPH 2026)*, 45(4), Article 121, 2026.
> DOI: 10.1145/3811386

If you are looking for the previous version of this library, it is still available in [this branch](https://github.com/MeshFEM/MeshFEM/tree/legacy-public-master). The `README` there describes some of the legacy C++ binaries, e.g., implementing periodic homogenization for linear elasticity. While these binaries are still included, they are now deprecated. The recommended way of using this framework is through its existing Python bindings and its functionality for defining and generating bindings for new elements and energy types.

Functionality is now split across three main repositories:
- [MeshFEMCore](https://github.com/MeshFEM/MeshFEMCore) -- basic type definitions and common infrastructure.
- [MeshFEMSparse](https://github.com/MeshFEM/MeshFEMSparse) -- sparse matrix data structures, assembly routines, and direct solver interfaces; can be included in other projects wishing to use our [BlockCatamari](https://github.com/MeshFEM/BlockCatamari/) Cholesky code.
- [MeshFEM](https://github.com/MeshFEM) -- this repository; mesh data structures, element types, Python bindings, unit tests, and some legacy code.

Additionally, [MeshFEMDemos](https://github.com/MeshFEM/MeshFEMDemos) provides examples for users interested to see how new energy densities and elements can be defined. This demo repository automatically brings in MeshFEM when you build it, so it can also be a good starting point for working with our library. But be sure also to check out the demos in [python/demos](https://github.com/MeshFEM/MeshFEM/tree/master/python/demos) of this repository.

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

Dependencies for running the Jupyter notebooks: [see here](#running-the-jupyter-notebooks).

<a name='building'></a>
Obtaining and Building
----------------------
After installing the non-bundled dependencies, use the following commands on
Mac or Linux to download and build MeshFEM:

```
git clone https://github.com/jpanetta/MeshFEM
cd MeshFEM
mkdir build && cd build
cmake ..
make -j(# of jobs)
```

I would recommend instead using the [Ninja build system](https://ninja-build.org), which
means changing the last two lines to `cmake .. -GNinja` and `ninja`.

Running the Jupyter Notebooks
-----------------------------
Python bindings for parts of the MeshFEM codebase have been
generated using [pybind11](https://github.com/pybind/pybind11). They should already have been built
and installed in the `python` directory when you [built the main project](#user-content-building).

We include some example Jupyter notebooks to demonstrate some of the bound
functionality. Please follow the instructions below to get these notebooks and
the visualization code running.

### JuptyterLab and Extensions
To run the Jupyter notebooks, you will need to install JupyterLab and the `pythreejs` library.
Ideally you would use [my fork](https://github.com/jpanetta/pythreejs) `pythreejs`, which provides
additional features, but this has become harder to install recently due to updates to the JupyterLab
ecostystem. While installing my version is still possible using the instructions below, it's currently the main pain point in
getting started on a fresh machine.

We recommend that you install the Python dependencies and JupyterLab itself in a
virtual environment (e.g., with [venv](https://docs.python.org/3/library/venv.html)).

```bash
pip install wheel # Needed if installing in a virtual environment
# Recent versions of jupyterlab and related packages cause problems:
#   JupyerLab 3.4 and later has a bug where the tab and status bar GUI
#                 remains visible after taking a viewer fullscreen
#   ipykernel > 5.5.5 clutters the notebook with stdout content
#   ipywidgets 8 and juptyerlab-widgets 3.0 break pythreejs
pip install jupyterlab==3.3.4 ipykernel==5.5.5 ipywidgets==7.7.2 jupyterlab-widgets==1.1.1
# If necessary, follow the instructions in the warnings to add the Python user
# bin directory (containing the 'jupyter' binary) to your PATH...

# See the note above about my pythreejs fork; if this is too much of a headache,
# you can do `pip install pythreejs` at the expense of losing some viewer features.
git clone https://github.com/jpanetta/pythreejs
cd pythreejs
export NODE_OPTIONS=--openssl-legacy-provider; # work around SSL errors in nodejs; see https://github.com/webpack/webpack/issues/14532#issuecomment-947012063
pip install -e .
cd js
jupyter labextension install .

pip install matplotlib scipy
```

Launch Jupyter lab from the root python directory:
```bash
cd python
jupyter lab
```

<!-- MarkdownTOC autolink="true" bracket="round" depth=3 -->
<!-- /MarkdownTOC -->

MeshFEM
=======

Installation
------------

Dependencies included directly as external projects:

- [json](https://github.com/nlohmann/json)
- [triangle](https://www.cs.cmu.edu/~quake/triangle.html)
- [tinyexpr](https://github.com/codeplea/tinyexpr)
- [Eigen](https://github.com/eigenteam/eigen-git-mirror)
- [TBB](https://github.com/01org/tbb), optional

Dependencies *not* included (should be installed system-wide):

- Boost
- Cholmod
- Umfpack

Elastic simulation
------------------

### Material file

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

#### Orthotropic material

    TODO

#### Anisotropic material

    TODO

### Boundary conditions file

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

**Tip**: Use `dirichletxy` to fix only the X and Y component of a region (then value[0:2] will be used). Same can be done with the other types.

Region box:

- `box`: use absolute coordinates.
- `box%`: relative to the bounding box of the input mesh.

**Units**:
- `mm` for node positions
- `N` for forces
- `MPa` for Young's modulus and traction (same as `N/mm^2`)

#### Examples

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


### Running the simulation

    ./Simulate_cli -m B9Creator.material -b loads.bc -o output.msh <input_mesh>

The only possible output file format is `.msh`.

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
- `Ku`: per-vertex actual force applied to the shape (including `no_rigid_motion` compensation, and ignoring external forces on Dirichlet nodes).
- `strain`: per-element strain tensor.
- `stress`: per-element stress tensor.

**Note**: per-vertex vector attributes (displacements `u` or `load`) are always stored as `Vector3d`, even in 2D (in which case they are padded with 0). Similarly, `strain` and `stress` tensors are stored as always `3x3` matrices, possibly padded with 0 (for the 2D case).

### Post-processing

To interpret the results of a simulation, the `tools/msh_processor` can be used. E.g.:

    $MeshFEM/build/tools/msh_processor in.msh -e ‘stress’ --eigenvalues --max --max

Will compute the maximum max eigenvalue of the stress field.

    $MeshFEM/build/tools/msh_processor in.msh -e ‘stress’ --eigenvalues --max -o out.msh

Will write a file `out.msh` with a scalar field with the max stress of each element.


Homogenization
--------------

Given a base material, you can homogenize the behavior of a periodic pattern (2D or 3D) in a linearly deformed cell (square in 2D, cube in 3D) by calling the following code:

    DeformedCells_cli examples/meshes/square_hole.off -m $MICRO_DIR/materials/B9Creator.material --homogenize --jacobian '2 0 0 1' --transformVersion

Explanation of the arguments:

- ` --jacobian 'xx xy yx yy'`. This is the Jacobian of the deformation that maps the undeformed square/cube to the deformed configuration.

- `--transformVersion`.
By default, `DeformedCells_cli` actually warps the periodic mesh into a parallelogram and then runs homogenization. If you pass `--transformVersion` it solves the transformed homogenization problem over the original undeformed mesh (by transforming the printing material properties accordingly and transforming the resulting effective tensor back). Both approaches should give identical results up to roundoff.

- It has an additional mode for the 2D case (`--parametrizedTransform`) where a sequence of deformations are read from stdin (one per line) and the resulting tensor is output for each. These deformations are parametrized by `theta lambda`, which specifies the Jacobian $$J = Rot(\theta) [\lambda\; 0; 0\; 1] Rot(\theta)^T$$.

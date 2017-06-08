MeshFEM
=======

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

**Tip**: Use `dirichletxy` to fix only the X and Y component of a region (then value[0:2] will be used). Same can done with the other types.

Region box:

- `box`: use absolute coordinates.
- `box%`: relative to the bounding box of the input mesh.

**Units**:
- `mm` for node positions
- `N` for forces
- `MPa` for Young's modulus and traction (same as `N/mm^2`)

### Run the simulation

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

**Note**: per-vertex vector attributes (displacements `u` or `load`) are always stored as `Vector3d`, even in 2D (in which case they are padded with 0). Similarly, `strain` and `stress` tensors are stored as always `3x3` matrices, possibly padde with 0 (for the 2D case).
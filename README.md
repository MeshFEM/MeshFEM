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

### Run the simulation

    ./Simulate_cli -m B9Creator.material -b loads.bc -o output.msh <input_mesh>

Accepted file formats:
- `input.msh`: tet-mesh only (no triangles)
- ???
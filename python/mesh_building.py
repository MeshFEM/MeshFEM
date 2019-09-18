import mesh

def get2DMesh(mesh_path, degree=2):
    if degree == 1: 
        return mesh.LinearFEM2DMesh(mesh_path)
    elif degree == 2:
        return mesh.QuadraticFEM2DMesh(mesh_path)
    return None

def get3DMesh(mesh_path, degree=2):
    if degree == 1:
        return mesh.LinearFEM3DMesh(mesh_path)
    elif degree == 2:
        return mesh.QuadraticFEM3DMesh(mesh_path)
    return None

def getMesh(mesh_path, dimension, degree=2):
    """ Return a mesh of the given dimension and finite element method degree.
    The mesh is created from the given path. 
    
    Arguments:
        mesh_path -- The path to the mesh. The supported file formats are those supported
            by the MeshFEM library, those format are listed in the file
            MeshFEM/src/lib/MeshFEM/MeshIO.hh in the Format enumeration.
        dimension -- The dimension of the mesh. Must be either 2 or 3.
        degree -- The finite element method degree of the mesh. Must be either 1 for linear or
            2 for quadratic.
    """
    
    if dimension != 2 and dimension != 3:
        raise ValueError(f'dimension must be either 2 or 3 not {dimension}')
    if degree != 1 and degree != 2:
        raise ValueError(f'degree must be either 1 or 2 no {degree}')
        
    if dimension == 2:
        return get2DMesh(mesh_path, degree)
    return get3DMesh(mesh_path, degree)
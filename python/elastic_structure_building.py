import elastic_structure
import energy

def getElasticStructureTypeName(e, m):
    return (('Linear' if m.degree == 1 else 'Quadratic')
        + 'FEM'
        + ('2D' if m.dimension == 2 else '3D')
        + ('LinearElastic' if e.type == energy.EnergyType.LINEAR else 'NeoHookean')
        + 'ElasticStructure')

def getElasticStructure(e, m):
    """ Return an elastic structure for a given mesh that
    is made of elastic material with the given energy.
    
    The mesh and the energy must have the same dimension.
    
    Arguments:
        e -- The energy.
        m -- The mesh.
    """

    return eval(f'elastic_structure.detail.{getElasticStructureTypeName(e, m)}')(e, m)

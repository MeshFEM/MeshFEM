################################################################################
#  auto-generated from @PROJECT_SOURCE_DIR@/python/init_template.py
################################################################################
import sys as _sys
_sys.path.insert(0, '@PROJECT_SOURCE_DIR@/python')
_sys.path.insert(0, '@PROJECT_SOURCE_DIR@/3rdparty/OffscreenRenderer/python')

import os as _os
if hasattr(_os, 'add_dll_directory'):
    for _d in ['@PROJECT_SOURCE_DIR@/python']:
        if _os.path.isdir(_d):
            _os.add_dll_directory(_d)

import sparse_matrices
import energy
import mesh
import parallelism
from mesh import Mesh, PeriodicCondition

import importlib.util
if importlib.util.find_spec('pythreejs') is not None:
    import tri_mesh_viewer

import elastic_solid

class EmbeddedMesh:
    """
    Wrapper to support using `sim_utils` and viewer on an embedding of a
    mesh `m` specified by optimization variables `embeddingVars`
    """
    def __init__(self, m, embeddingVars, embeddingDimension = None):
        self.dimension = embeddingVars.numVars() // m.numNodes() if embeddingDimension is None else embeddingDimension
        self.m_mesh = m
        self.evars = embeddingVars
        self.numVertices = m.numVertices()
        if self.evars.numVars() != m.numNodes() * self.dimension:
            raise Exception('Unexpected variable size')
        self.vis_mesh = mesh.Mesh(self.embeddedVertices(), m.elements())
    def mesh(self): return self.m_mesh
    def embeddedVertices(self):
        return self.evars.getVars().reshape((-1, self.dimension))[:self.numVertices]
    def visualizationGeometry(self, normalCreaseAngle, *args, **kwargs):
        self.vis_mesh.setVertices(self.embeddedVertices())
        return self.vis_mesh.visualizationGeometry(normalCreaseAngle=normalCreaseAngle, *args, **kwargs)
    def visualizationField(self, *args, **kwargs):
        return self.vis_mesh.visualizationField(*args, **kwargs)

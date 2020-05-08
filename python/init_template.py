################################################################################
#  auto-generated from @PROJECT_SOURCE_DIR@/python/init_template.py
################################################################################
import sys as _sys
_sys.path.insert(0, '@PROJECT_SOURCE_DIR@/python')

import sparse_matrices
import energy
import mesh
import parallelism
from mesh import Mesh, PeriodicCondition

from energy_building import *

import importlib
if importlib.util.find_spec('pythreejs') is not None:
    import tri_mesh_viewer

# may not have elastic_structure build if Boost is not presented
try:
    import elastic_structure
except:
    pass

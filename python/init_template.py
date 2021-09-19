################################################################################
#  auto-generated from @PROJECT_SOURCE_DIR@/python/init_template.py
################################################################################
import sys as _sys
_sys.path.insert(0, '@PROJECT_SOURCE_DIR@/python')
_sys.path.insert(0, '@PROJECT_SOURCE_DIR@/3rdparty/OffscreenRenderer/python')

import sparse_matrices
import mesh
import parallelism
from mesh import Mesh, PeriodicCondition

import importlib
import importlib.util
if importlib.util.find_spec('pythreejs') is not None:
    import tri_mesh_viewer

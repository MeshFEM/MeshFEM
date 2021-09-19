from _parallelism import *
import os as _os

if 'MESHFEM_NUM_THREADS' in _os.environ:
    cpus = int(_os.environ['MESHFEM_NUM_THREADS'])
    set_max_num_tbb_threads(cpus)

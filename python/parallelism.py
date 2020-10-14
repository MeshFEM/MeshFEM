from _parallelism import *
import os as _os

if 'SLURM_CPUS_PER_TASK' in _os.environ:
    set_max_num_tbb_threads(int(_os.environ['SLURM_CPUS_PER_TASK']))

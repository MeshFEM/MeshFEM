'''
Python Helper Functions for benchmarking Parametrization using MeshFEM's new feature `MeshEnergy`

Author:  Xinzhuo (johnson) Hu
Created: 01/11/2025  2:07:55
'''

import os, sys
sys.path.append('../')
import subprocess
import MeshFEM
import mesh, mesh_energy, energy
import parametrization, py_newton_optimizer, benchmark, flip_avoiding_step_length
import tinyad_parametrization, dirichlet_demo
# import numpy as np
import copy, time
# import igl
import csv

from typing import NamedTuple

import pickle
from pathlib import Path
from typing import Any, Dict, Mapping

def save_dict(d: Mapping[str, Any], file: Path, compressed: bool = False) -> None:
    """Save a mapping as a pickle (optionally gzip-compressed)."""
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)

    if compressed or file.suffix == ".gz":
        import gzip
        with gzip.open(file, "wb") as f:
            pickle.dump(dict(d), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(file, "wb") as f:
            pickle.dump(dict(d), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(file: Path) -> Dict[str, Any]:
    """Load and return a dict from a pickle (supports .gz)."""
    file = Path(file)
    if file.suffix == ".gz":
        import gzip
        with gzip.open(file, "rb") as f:
            return pickle.load(f)
    else:
        with open(file, "rb") as f:
            return pickle.load(f)

def map_vertices_to_circle_area_normalized(V, F, bnd):
    import igl
    import numpy as np
    """
    Python equivalent of the C++ function:

        void map_vertices_to_circle_area_normalized(
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            const Eigen::VectorXi& bnd,
            Eigen::MatrixXd& UV)

    Parameters
    ----------
    V : (n, 3) float ndarray
        Vertex positions
    F : (m, 3) int ndarray
        Triangle indices
    bnd : (k,) int ndarray
        Boundary vertex indices

    Returns
    -------
    bc : (k, 2) float ndarray
        UV coordinates for the boundary vertices, placed on a circle
        whose radius is sqrt(mesh_area / pi).
    """
    # 1) Compute total mesh area via doublearea
    #    igl.doublearea(...) returns one "double area" value per face
    dblArea_orig = igl.doublearea(V, F)  # shape (m,)
    area = dblArea_orig.sum() / 2.0
    radius = np.sqrt(area / np.pi)

    # Uncomment if you want the same console output as in C++:
    # print(f"map_vertices_to_circle_area_normalized, area = {area}, radius = {radius}")
    map_ij = np.zeros((V.shape[0], ), dtype=int)
    interior = []
    isOnBnd = np.zeros((V.shape[0], ), dtype=bool)
    for i in range(bnd.shape[0]):
        isOnBnd[bnd[i]] = True
        map_ij[bnd[i]] = i
    for i in range(isOnBnd.shape[0]):
        if (not isOnBnd[i]):
            map_ij[i] = len(interior)
            interior.append(i)

    # 2) Build a running length array along boundary vertices
    k = bnd.shape[0]
    length = np.zeros(k)
    for i in range(1, k):
        prev_idx = bnd[i - 1]
        curr_idx = bnd[i]
        length[i] = length[i - 1] + np.linalg.norm(V[prev_idx] - V[curr_idx])

    # Add the distance between the last and the first boundary vertex
    total_len = length[-1] + np.linalg.norm(V[bnd[0]] - V[bnd[-1]])

    # 3) Place boundary vertices along the circle of computed radius
    bc = np.zeros((k, 2))
    for i in range(k):
        frac = length[i] * (2.0 * np.pi) / total_len
        bc[map_ij[bnd[i]], 0] = radius * np.cos(frac)
        bc[map_ij[bnd[i]], 1] = radius * np.sin(frac)
        # bc[i, 0] = radius * np.cos(frac)
        # bc[i, 1] = radius * np.sin(frac)
    return bc


def getBDdataOnUnitCircle(m):
    import igl
    BV = m.boundaryVertices()
    bloop = m.boundaryLoops()[0][::-1]
    bdry_uv = igl.map_vertices_to_circle(m.vertices(), BV[bloop])
    bdry_uv[bloop] =  bdry_uv.copy()
    return bdry_uv

def getBDdataOnNormalizedCircle(m):
    import igl
    import numpy as np
    BV = m.boundaryVertices()
    bnd_loop = igl.boundary_loop(m.elements())
    bloop = np.searchsorted(BV, bnd_loop)
    bdry_uv = map_vertices_to_circle_area_normalized(m.vertices(), m.elements(), bnd_loop)
    bdry_uv[bloop] =  bdry_uv.copy()
    return bdry_uv

# read mesh and scale down vertices
def read_mesh(mesh_path : str):
    import param_utils
    import numpy as np
    m_ori = param_utils.load(mesh_path) # supports loading, e.g., `input.msh.xz`
    vertices_ori = m_ori.vertices()
    elements_ori = m_ori.elements()
    mesh_area_ori = (m_ori.elementVolumes()).sum()
    vertices_scale_down = vertices_ori / np.sqrt(mesh_area_ori)
    return mesh.Mesh(vertices_scale_down, elements_ori)

def tutteInitialization(m, bdry_uv):
    # Tutte Initialization
    uv_init = parametrization.harmonic(m, bdry_uv, False)
    flip_list = parametrization.getFlips(m, uv_init)
    if len(flip_list) > 0:  uv_init = parametrization.harmonic(m, bdry_uv, True)
    return uv_init

def delete_all_files_in_folder(folder_path):
    """
    Deletes only regular files (and symlinks) in the given folder,
    leaving subdirectories (and their contents) untouched.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # If it's a file or a symlink, remove it
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)

def delete_txt_files(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def processUVTXTs(source_path, to_path, txt_file_prefix_str, index_offset=0):
    import numpy as np
    """
    Processes txt files in a given folder, converting them to raveled NumPy arrays
    and saving them as compressed .npz files. Deletes the original txt files after
    ensuring the same number of .npz files are created.

    Args:
        source_path (str): Path to the folder containing the txt files.
        to_path (str): Path to the folder containing the npz files.
        txt_file_prefix_str: e.g., "uv_Eigen_Iter_"

    Returns:
        bool: True if the number of .npz files matches the original txt files, False otherwise.
    """
    # Get all files in the folder
    txt_files = [f for f in os.listdir(source_path) if f.startswith(txt_file_prefix_str) and f.endswith(".txt")]
    npz_files_created = 0

    for txt_file in txt_files:
        try:
            # Build full path for the txt file
            txt_path = os.path.join(source_path, txt_file)
            
            # Read the txt file into a NumPy array
            data = np.loadtxt(txt_path)
            
            # Ensure the data has two columns
            if data.ndim == 1 or data.shape[1] != 2:
                raise ValueError(f"File {txt_file} does not have two columns.")
            
            # Get the raveled version of the array
            raveled_data = data.ravel()
            
            # Construct the .npz file name
            file_index = int(txt_file.split('_')[-1].split('.')[0]) + index_offset  # Extract i from "uv_Eigen_Iter_i.txt"
            npz_file_name = f"uv_ravel_iter_{file_index}.npz"
            npz_path = os.path.join(to_path, npz_file_name)
            
            # Save the raveled array to a compressed .npz file
            np.savez_compressed(npz_path, arr=raveled_data)
            npz_files_created += 1
        except Exception as e:
            print(f"Error processing file {txt_file}: {e}")

    # Check if the number of .npz files matches the number of txt files
    npz_files = [f for f in os.listdir(to_path) if f.startswith("uv_ravel_iter_") and f.endswith(".npz")]
    if len(npz_files) == len(txt_files):
        # If numbers match, delete the txt files
        for txt_file in txt_files:
            try:
                os.remove(os.path.join(source_path, txt_file))
            except Exception as e:
                print(f"Error deleting file {txt_file}: {e}")
        return True
    else:
        print(f"Mismatch in file counts: {len(txt_files)} txt files vs {len(npz_files)} npz files.")
        return False

def parse_custom_csv(folder_path, csv_filename):  
    import numpy as np
    """
    For processing Roi's Composite Majorization's stats recording csv file
    """
    csv_path = os.path.join(folder_path, csv_filename)

    # Initialize storage
    table_data = {}
    summary_stats = {}

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    # Identify the split between table and summary
    empty_line_index = next(i for i, row in enumerate(lines) if len(row) == 0)

    # Parse table header
    header = lines[0]
    for key in header:
        table_data[key] = []

    # Parse table data
    for row in lines[1:empty_line_index]:
        for key, value in zip(header, row):
            table_data[key].append(float(value))

    # Convert to numpy arrays
    for key in table_data:
        table_data[key] = np.array(table_data[key])

    # Parse summary statistics
    for row in lines[empty_line_index+1:]:
        if len(row) == 2:
            key, value = row
            summary_stats[key.strip()] = float(value.strip())

    return table_data, summary_stats

# self-defined NamedTuple for hessian arrays
class HessianStats(NamedTuple):
    import numpy as np
    from numpy.typing import NDArray
    projected:  NDArray[np.int_]
    shifted:    NDArray[np.float_]
    indefinite: NDArray[np.int_]

def runSYDParam(m, ProjectionStrategy, EigenvalueModification, 
                ProjectionType, AutodiffSetting, SteplengthComputer,
                max_iter=200, hessian_shift=1e-12, grad_tol=None, uvsave_path=None):
    import numpy as np
    obj_history = []
    time_history = []
    grad_norm_history = []
    hessian_projected_history = []
    hessian_shifted_amount_history = []
    step_norm_history = []
    directional_derivative_history = []

    def customCallback(prob, i):
        obj_history.append(prob.energy())
        grad_norm_history.append(np.linalg.norm(prob.gradient()))
        # Record hessian_projected and hessian_shifted_amount_history as well in customCallback, assume recording time is less significant
        if i > 1:  
            hessian_projected_history.append(int(prob.hessianWasProjected))
            hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)
        time_history.append(-benchmark.totalTime('Callback$') + time.perf_counter())
    
    def customSaveUVCallback(prob, i):
        obj_history.append(prob.energy())
        grad_norm_history.append(np.linalg.norm(prob.gradient()))
        if i > 1:  
            hessian_projected_history.append(int(prob.hessianWasProjected))
            hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)
        # save UV in compressed mode
        uv_fn = 'uv_ravel_'+ 'iter_' + str(i-1)
        uv_arr = uv.getVars()
        np.savez_compressed(os.path.join(uvsave_path, uv_fn), arr=uv_arr)
        time_history.append(-benchmark.totalTime('Callback$') + time.perf_counter())
    
    def customSaveStepDCallback(prob, step, directional_derivative):
        step_norm_history.append(np.linalg.norm(step))
        directional_derivative_history.append(-directional_derivative)

    uv = mesh_energy.NodalVars(m, 2)
    bdry_uv = getBDdataOnNormalizedCircle(m)
    uv_init = tutteInitialization(m, bdry_uv)
    uv.setVars(uv_init.ravel())

    # Configuring User Options 
    # AutodiffSetting
    if AutodiffSetting == 'AD':  symmdiri_energy = energy.SymmetricDirichletDerivativeFree(2)
    elif AutodiffSetting == 'NoAD': symmdiri_energy = energy.SymmetricDirichlet(2)
    else:  raise NameError(f"[runSYDParam] MeshFEM Solver Configuration: Autodiff Setting {AutodiffSetting} is not implemented.")

    # EigenvalueModification
    if EigenvalueModification == 'Clamp': symmdiri_energy.useAbsProjection = False
    elif EigenvalueModification == 'Abs': symmdiri_energy.useAbsProjection = True
    else: raise NameError(f"[runSYDParam] MeshFEM Solver Configuration: Eigenvalue Modification {EigenvalueModification} is not implemented.")

    # Construct `SymmetricDirichlet` parametrization energy and problem
    param = mesh_energy.Parametrization(m, uv, symmdiri_energy)
    prob = py_newton_optimizer.NewtonMultiobjectiveProblem(uv, [param])

    # Step Length Computer
    if SteplengthComputer == 'FlipAvoid':
        prob.initialFeasibleStepLengthComputer = flip_avoiding_step_length.FlipAvoidingStepLength(m.elements())
        prob.initialFeasibleStepLengthComputer.backoffFactor = 0.8    # in accordance to Composite Majorization
    elif SteplengthComputer == 'NoFlipAvoid':
        pass
    else:  raise NameError(f"[runSYDParam] MeshFEM Solver Configuration: Steplength Computer {SteplengthComputer} is not implemented.")

    if uvsave_path is None:  prob.setCustomIterationCallback(customCallback)
    else:
        if not os.path.exists(uvsave_path):
            raise RuntimeError(f"[Error] The uv_save path: {uvsave_path} does not exist!")
        prob.setCustomIterationCallback(customSaveUVCallback)
    prob.setCustomLineSearchBeganCallback(customSaveStepDCallback)

    # Projection Type
    if ProjectionType == 'FBased':  param.useXBasedProjection = False
    elif ProjectionType == 'XBased': param.useXBasedProjection = True
    else:  raise NameError(f"[runSYDParam] MeshFEM Solver Configuration: Projection Type {ProjectionType} is not implemented.")

    # Work around energy nullspace by adding a small shift
    prob.hessianShift = hessian_shift
    print("-------------------------------------------------------------------------")
    print(f'[SymDiriParam] MeshFEM Problem: Set Hessian Shift to {hessian_shift}.')
    print("-------------------------------------------------------------------------")
    opt = prob.optimizer()
    opt.options.niter = max_iter
    if grad_tol is not None: opt.options.gradTol = grad_tol  # default is 2e-8

    # Projection Strategy
    if ProjectionStrategy == 'Adaptive':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAdaptive()
        opt.options.hessianProjectionController.numConsecutiveIndefiniteStepsBeforeEnable = 0
        opt.options.hessianProjectionController.numProjectionStepsBeforeDisable = 2
    elif ProjectionStrategy == 'Always':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()
    elif ProjectionStrategy == 'Never':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionNever()
    else:  raise NameError(f"[runSYDParam] MeshFEM Solver Configuration: Projection Strategy {ProjectionStrategy} is not implemented.")
    
    # Run Optimization
    benchmark.reset()
    start_time = time.perf_counter()
    cr = opt.optimize()
    # benchmark.report()

    hessian_projected_history.append(int(prob.hessianWasProjected)) # The projection status of the Hessian used in are i-1
    hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)
    # Also saving hessian_related information (arrays)
    hessian_projected_arr = np.array(hessian_projected_history, dtype=int)
    hessian_shifted_arr = np.array(hessian_shifted_amount_history, dtype=float)
    hessian_indef_arr = np.array(cr.indefinite, dtype=int)

    if uvsave_path is not None:     
        bk_dict = benchmark.to_dict() 
        obj_arr = np.array(obj_history)
        time_arr = np.array(time_history) - start_time
        grad_norm_arr = np.array(grad_norm_history)
        # we saved uv coordinates per-iteration and hessian_projected_history
        step_size_arr = np.array(step_norm_history)
        dd_arr = np.array(directional_derivative_history)

        obj_filename = 'obj_history.npy'
        time_filename = 'time_history.npy'
        grad_norm_filename = 'grad_norm_history.npy'
        hp_filename = 'hessian_projected_history.npy'
        hs_filename = 'hessian_shifted_amount_history.npy'
        hindef_filename = 'hessian_indefinite_history.npy'
        step_filename = 'step_size_history.npy'
        dd_filename = 'directional_derivative_history.npy'
        benchmark_filename = 'benchmark_dict.pkl'

        np.save(os.path.join(uvsave_path, obj_filename), obj_arr)
        np.save(os.path.join(uvsave_path, time_filename), time_arr)
        np.save(os.path.join(uvsave_path, grad_norm_filename), grad_norm_arr)
        np.save(os.path.join(uvsave_path, hp_filename), hessian_projected_arr)
        np.save(os.path.join(uvsave_path, hs_filename), hessian_shifted_arr)
        np.save(os.path.join(uvsave_path, hindef_filename), hessian_indef_arr)
        np.save(os.path.join(uvsave_path, step_filename), step_size_arr)
        np.save(os.path.join(uvsave_path, dd_filename), dd_arr)
        save_dict(bk_dict, os.path.join(uvsave_path, benchmark_filename))
        print(f"[File] Saved UV '.npz' files, {obj_filename}, {time_filename}, {grad_norm_filename}, {hp_filename}, {hs_filename}, {hindef_filename}, {step_filename}, {dd_filename}, {benchmark_filename} in {uvsave_path}.")
    else:
        bk_dict = benchmark.to_dict()
        time_arr = np.array(time_history) - start_time
        hessian_stats = HessianStats(
            projected=hessian_projected_arr,
            shifted=hessian_shifted_arr,
            indefinite=hessian_indef_arr,
        )
        return np.array(obj_history), time_arr, np.array(grad_norm_history), bk_dict, hessian_stats


def runSYDParam_matchBaseline(m, baseline_str, max_iter=200, grad_tol=2e-8, clamp_eps=1e-9, uvsave_path=None):
    import numpy as np
    '''
    function to perform symmetric dirichlet parameterization using MeshFEM matching two baseline methods: CM and TinyAD
    baseline_str: 
    'MeshFEM_CM':           MeshFEM Matching CompMajor, Always F-based projection, Flipavoid linesearch, per-element hessian shift = 1e-6
    'MeshFEM_TAD_Fad':      MeshFEM Matching TinyAD, Always X-based projection, normal linesearch, Hessian clamping value = 1e-9, Standard F-autodiff
    'MeshFEM_TAD_Xad':      MeshFEM Matching TinyAD but using x-autodiff using the same (inefficient) formulas from the TinyAD demo
    '''
    # Tutte Initialization
    obj_history = []
    time_history = []
    grad_norm_history = []
    hessian_projected_history = []
    hessian_shifted_amount_history = []
    step_norm_history = []
    directional_derivative_history = []

    def customCallback(prob, i):
        obj_history.append(prob.energy())
        grad_norm_history.append(np.linalg.norm(prob.gradient()))
        # Record hessian_projected and hessian_shifted_amount_history as well in customCallback, assume recording time is less significant
        if i > 1:  
            hessian_projected_history.append(int(prob.hessianWasProjected))
            hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)
        time_history.append(-benchmark.totalTime('Callback$') + time.perf_counter())
    
    def customSaveUVCallback(prob, i):
        obj_history.append(prob.energy())
        grad_norm_history.append(np.linalg.norm(prob.gradient()))
        if i > 1:  
            hessian_projected_history.append(int(prob.hessianWasProjected))
            hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)
        # save UV in compressed mode
        uv_fn = 'uv_ravel_'+ 'iter_' + str(i-1)
        uv_arr = uv.getVars()
        np.savez_compressed(os.path.join(uvsave_path, uv_fn), arr=uv_arr)
        time_history.append(-benchmark.totalTime('Callback$') + time.perf_counter())
    
    def customSaveStepDCallback(prob, step, directional_derivative):
        step_norm_history.append(np.linalg.norm(step))
        directional_derivative_history.append(-directional_derivative)

    uv = mesh_energy.NodalVars(m, 2)
    bdry_uv = getBDdataOnNormalizedCircle(m)
    uv_init = tutteInitialization(m, bdry_uv)
    uv.setVars(uv_init.ravel())

    if baseline_str == 'MeshFEM_TAD_Fad':
        symmdiri_energy = energy.SymmetricDirichletDerivativeFree(2)  # AutoDifferentiate
        symmdiri_energy.useAbsProjection = False  # Clamp
        param = mesh_energy.Parametrization(m, uv, symmdiri_energy)
        param.useXBasedProjection = True  # X-based
        param.xBasedProjectionClampEps = clamp_eps
        param.elementHessianShift = 0
    elif baseline_str == 'MeshFEM_TAD_Xad':
        param = dirichlet_demo.param_symdirichlet_element_tad_compare(m, uv)
        param.useXBasedProjection = True  # X-based
        param.xBasedProjectionClampEps = clamp_eps
        param.elementHessianShift = 0
    elif baseline_str in ['MeshFEM_CM', 'MeshFEM_CM_adp']:
        symmdiri_energy = energy.SymmetricDirichlet(2) # Analytical Derivatives
        symmdiri_energy.useAbsProjection = False # Clamp
        param = mesh_energy.Parametrization(m, uv, symmdiri_energy)
        param.useXBasedProjection = False  # F-based
        param.elementHessianShift = 1e-6 
    else:  raise RuntimeError(f"[MeshFEM Matching Baseline] {baseline_str} is not a valid option!")

    prob = py_newton_optimizer.NewtonMultiobjectiveProblem(uv, [param])
    if baseline_str in ['MeshFEM_CM', 'MeshFEM_CM_adp']:
        prob.initialFeasibleStepLengthComputer = flip_avoiding_step_length.FlipAvoidingStepLength(m.elements())
        prob.initialFeasibleStepLengthComputer.backoffFactor = 0.8    # in accordance to Composite Majorization

    if uvsave_path is None:  prob.setCustomIterationCallback(customCallback)
    else:
        if not os.path.exists(uvsave_path):
            raise RuntimeError(f"[Error] The uv_save path: {uvsave_path} does not exist!")
        prob.setCustomIterationCallback(customSaveUVCallback)
    prob.setCustomLineSearchBeganCallback(customSaveStepDCallback)
    prob.hessianShift = 0

    # opt parameter set up 
    opt = prob.optimizer()
    opt.options.niter = max_iter
    opt.options.gradTol = grad_tol
    if baseline_str == 'MeshFEM_CM_adp':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAdaptive() #COP2 Adaptive projection
        opt.options.hessianProjectionController.numConsecutiveIndefiniteStepsBeforeEnable = 0
        opt.options.hessianProjectionController.numProjectionStepsBeforeDisable = 2
    else:
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()  # Always projection
    if baseline_str in ['MeshFEM_TAD_Fad', 'MeshFEM_TAD_Xad']:
        # match linesearch parameters of TinyAD
        opt.options.backtrack_shrink_factor = 0.8
        opt.options.nbacktrack_iter = 64

    # Run Optimization
    benchmark.reset()
    start_time = time.perf_counter()
    cr = opt.optimize()

    hessian_projected_history.append(int(prob.hessianWasProjected)) # The projection status of the Hessian used in are i-1
    hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)
    # Also saving hessian_related information (arrays)
    hessian_projected_arr = np.array(hessian_projected_history, dtype=int)
    hessian_shifted_arr = np.array(hessian_shifted_amount_history, dtype=float)
    hessian_indef_arr = np.array(cr.indefinite, dtype=int)

    if uvsave_path is not None:      
        obj_arr = np.array(obj_history)
        time_arr = np.array(time_history) - start_time
        grad_norm_arr = np.array(grad_norm_history)
        # we saved uv coordinates per-iteration and hessian_projected_history
        step_size_arr = np.array(step_norm_history)
        dd_arr = np.array(directional_derivative_history)
        bk_dict = benchmark.to_dict()

        obj_filename = 'obj_history.npy'
        time_filename = 'time_history.npy'
        grad_norm_filename = 'grad_norm_history.npy'
        hp_filename = 'hessian_projected_history.npy'
        hs_filename = 'hessian_shifted_amount_history.npy'
        hindef_filename = 'hessian_indefinite_history.npy'
        step_filename = 'step_size_history.npy'
        dd_filename = 'directional_derivative_history.npy'
        benchmark_filename = 'benchmark_dict.pkl'

        np.save(os.path.join(uvsave_path, obj_filename), obj_arr)
        np.save(os.path.join(uvsave_path, time_filename), time_arr)
        np.save(os.path.join(uvsave_path, grad_norm_filename), grad_norm_arr)
        np.save(os.path.join(uvsave_path, hp_filename), hessian_projected_arr)
        np.save(os.path.join(uvsave_path, hs_filename), hessian_shifted_arr)
        np.save(os.path.join(uvsave_path, hindef_filename), hessian_indef_arr)
        np.save(os.path.join(uvsave_path, step_filename), step_size_arr)
        np.save(os.path.join(uvsave_path, dd_filename), dd_arr)
        save_dict(bk_dict, os.path.join(uvsave_path, benchmark_filename))
        print(f"[File] Saved UV '.npz' files, {obj_filename}, {time_filename}, {grad_norm_filename}, {hp_filename}, {hs_filename}, {hindef_filename}, {step_filename}, {dd_filename}, {benchmark_filename} in {uvsave_path}.")
    else:
        bk_dict = benchmark.to_dict()
        time_arr = np.array(time_history) - start_time
        hessian_stats = HessianStats(
            projected=hessian_projected_arr,
            shifted=hessian_shifted_arr,
            indefinite=hessian_indef_arr,
        )
        return np.array(obj_history), time_arr, np.array(grad_norm_history), bk_dict, hessian_stats


def runSymmds_TinyAD(m, max_iter=200, grad_tol=2e-8, uvsave_path=None):
    import numpy as np
    bdry_uv = getBDdataOnNormalizedCircle(m)
    uv_init = tutteInitialization(m, bdry_uv)

    benchmark.reset()
    if uvsave_path is not None:
        uv_opt, obj_history, grad_history, time_history, step_size_history, dd_history = tinyad_parametrization.symmdsParamTinyAD(m, uv_init, max_iter, grad_tol, True, uvsave_path)
        bk_dict = benchmark.to_dict()
        # process all saved txt files into compressed npz files
        if not processUVTXTs(uvsave_path, uvsave_path, "uv_Eigen_Iter_"):  raise RuntimeError(f"[Error] In Process Eigen txts in {uvsave_path}.")
        obj_arr = np.array(obj_history)
        time_arr = np.array(time_history)
        grad_norm_arr = np.array(grad_history)
        step_size_arr = np.array(step_size_history)
        dd_arr = np.array(dd_history)

        obj_filename = 'obj_history.npy'
        time_filename = 'time_history.npy'
        grad_norm_filename = 'grad_norm_history.npy'
        step_filename = 'step_size_history.npy'
        dd_filename = 'directional_derivative_history.npy'
        benchmark_filename = 'benchmark_dict.pkl'

        np.save(os.path.join(uvsave_path, obj_filename), obj_arr)
        np.save(os.path.join(uvsave_path, time_filename), time_arr)
        np.save(os.path.join(uvsave_path, grad_norm_filename), grad_norm_arr)
        np.save(os.path.join(uvsave_path, step_filename), step_size_arr)
        np.save(os.path.join(uvsave_path, dd_filename), dd_arr)
        save_dict(bk_dict, os.path.join(uvsave_path, benchmark_filename))
        print(f"[File] Saved UV '.npz' files, {obj_filename}, {time_filename}, {grad_norm_filename}, {step_filename}, {dd_filename}, {benchmark_filename} in {uvsave_path}.")
    else:
        uv_opt, obj_history, grad_history, time_history, step_size_history, dd_history  = tinyad_parametrization.symmdsParamTinyAD(m, uv_init, max_iter, grad_tol, False)
        bk_dict = benchmark.to_dict()
        return np.array(obj_history), np.array(time_history), np.array(grad_history), bk_dict

def runSLIM(model_name, model_path, thread_num=0, uvsave_path=None):
    threads_str = "OMP_NUM_THREADS="
    exe_binary_str = "./ReweightedARAP"
    model_uv_name = model_name + "_slim_uv.off"
    # create TEMP_FILE_PATH if it doesn't exists
    UV_FILE_PATH = "SLIM_TEMP_UV"
    DATA_FILE_PATH = "SLIM_TEMP_DATA"
    if not os.path.exists(UV_FILE_PATH):  os.makedirs(UV_FILE_PATH)
    if not os.path.exists(DATA_FILE_PATH):  os.makedirs(DATA_FILE_PATH)

    if uvsave_path is not None: # SAVE UV AT EVERY ITERATION
        TEMP_FILE_PATH = UV_FILE_PATH
        model_uv_path = os.path.join(TEMP_FILE_PATH, model_uv_name)
        execute_str = threads_str + str(16) + " " + exe_binary_str + " " + model_path + " " + model_uv_path + " " + "yes"
        cmd = [
            # threads_str + str(16),
            exe_binary_str,
            model_path,
            model_uv_path,
            "yes"
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during execution: {e}")
            sys.exit(1)
        # Now we want to read data from SLIM_TEMP_UV
        import numpy as np
        obj_txt_path = os.path.join(TEMP_FILE_PATH, "obj_history.txt")
        obj_arr = np.loadtxt(obj_txt_path)
        grad_norm_txt_path = os.path.join(TEMP_FILE_PATH, "grad_norm_history.txt")
        grad_norm_arr = np.loadtxt(grad_norm_txt_path)
        obj_filename = 'obj_history.npy'
        grad_norm_filename = 'grad_norm_history.npy'
        np.save(os.path.join(uvsave_path, obj_filename), obj_arr)
        np.save(os.path.join(uvsave_path, grad_norm_filename), grad_norm_arr)
        # Process UVs
        if not processUVTXTs(TEMP_FILE_PATH, uvsave_path, "uv_SLIM_Iter_"):  raise RuntimeError(f"[Error] In Process SLIM txts in {TEMP_FILE_PATH}.")
        delete_all_files_in_folder(TEMP_FILE_PATH) # delete all files in TEMP_FILE_PATH
        print(f"[File] Saved UV '.npz' files, {obj_filename}, {grad_norm_filename} in {uvsave_path}.")
    else:
        TEMP_FILE_PATH = DATA_FILE_PATH
        model_uv_path = os.path.join(TEMP_FILE_PATH, model_uv_name)
        execute_str = threads_str + str(thread_num) + " " + exe_binary_str + " " + model_path + " " + model_uv_path + " " + "no"
        cmd = [
            # threads_str + str(thread_num),
            exe_binary_str,
            model_path,
            model_uv_path,
            "no"
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during execution: {e}")
            sys.exit(1)
        # Now we want to read data from SLIM_TEMP_DATA
        obj_txt_path = os.path.join(TEMP_FILE_PATH, "obj_history.txt")
        grad_norm_txt_path = os.path.join(TEMP_FILE_PATH, "grad_norm_history.txt")
        iter_time_txt_path = os.path.join(TEMP_FILE_PATH, "iter_time_history.txt")
        benchmark_data_txt_path = os.path.join(TEMP_FILE_PATH, "benchmark_data.txt")

        obj_arr = np.loadtxt(obj_txt_path)
        grad_norm_arr = np.loadtxt(grad_norm_txt_path)
        iter_time_arr = np.loadtxt(iter_time_txt_path)
        benchmark_data = np.loadtxt(benchmark_data_txt_path)

        # process iter_time_arr deduce initial parametrization time
        time_history_arr = iter_time_arr - iter_time_arr[0]
        # construct dictionary benchmark_dict
        benchmark_dict = {}
        benchmark_dict['totalTime'] = benchmark_data[0] - iter_time_arr[0]
        benchmark_dict['symbolic_fac_time'] = benchmark_data[1]
        benchmark_dict['numeric_fac_time'] = benchmark_data[2]
        benchmark_dict['hessian_eval_time'] = benchmark_data[3]
        benchmark_dict['linear_solve_time'] = benchmark_data[4]
        
        delete_all_files_in_folder(TEMP_FILE_PATH) # delete all files in TEMP_FILE_PATH
        return obj_arr, time_history_arr, grad_norm_arr, benchmark_dict

def runCompMajor(model_name, model_path, uvsave_path=None):
    exe_binary_str = "../../../CompMajor/build/CompMajor_bin"   # relative path of user xinzhuo on Julian's Linux Server
    model_out_name = model_name + "_out.obj"
    # create TEMP_FILE_PATH if it doesn't exists
    UV_FILE_PATH = "CompMajor_TEMP_UV"
    DATA_FILE_PATH = "CompMajor_TEMP_DATA"
    if not os.path.exists(UV_FILE_PATH):  os.makedirs(UV_FILE_PATH)
    if not os.path.exists(DATA_FILE_PATH):  os.makedirs(DATA_FILE_PATH)

    if uvsave_path is not None: # SAVE UV AT EVERY ITERATION
        model_out_save_path = os.path.join(UV_FILE_PATH, model_out_name)
        cmd = [
            exe_binary_str,
            model_path,
            model_out_save_path,
            str(200),
            str(1)
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during CompMajor parametrization (save UV) of {model_name}: {e}")
            sys.exit(1)
        
        # now we want to read data from CompMajor_TEMP_UV
        # read csv
        import numpy as np
        csv_file_name = model_out_name + "_timing.csv"
        table_data, summary_stats = parse_custom_csv(UV_FILE_PATH, csv_file_name)
        iter_time_arr = table_data["step_time"] + table_data["linesearch_time"]
        # process iter_time_arr based on step_time_arr
        iter_time_arr = np.pad(iter_time_arr, [(1, 0)])
        time_history_arr = np.cumsum(iter_time_arr)

        obj_filename = 'obj_history.npy'
        grad_norm_filename = 'grad_norm_history.npy'
        time_filename = 'time_history.npy'
        np.save(os.path.join(uvsave_path, obj_filename), table_data["objective_value"])
        np.save(os.path.join(uvsave_path, time_filename), time_history_arr)
        np.save(os.path.join(uvsave_path, grad_norm_filename), table_data["gradient_norm"])
        # Process UV TXTs
        txt_prefix = model_out_name + "_Iter_"
        if not processUVTXTs(UV_FILE_PATH, uvsave_path, txt_prefix, index_offset=1):  
            # CompMajor's UV Saving starts from 1 (not saving the tutte initialized mesh)
            raise RuntimeError(f"[Error] In Process CompMajor UV txts in {UV_FILE_PATH}.")
        delete_txt_files(UV_FILE_PATH)
        print(f"[File] Saved UV '.npz' files, {obj_filename}, {time_filename}, {grad_norm_filename} in {uvsave_path}.")
        return 
    else:
        model_out_save_path = os.path.join(DATA_FILE_PATH, model_out_name)
        cmd = [
            exe_binary_str,
            model_path,
            model_out_save_path,
            str(200),
            str(0)
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during CompMajor parametrization of {model_name}: {e}")
            sys.exit(1)
        
        # read csv
        csv_file_name = model_out_name + "_timing.csv"
        table_data, summary_stats = parse_custom_csv(DATA_FILE_PATH, csv_file_name)

        obj_arr = table_data["objective_value"]
        grad_norm_arr = table_data["gradient_norm"]
        iter_time_arr = table_data["step_time"] + table_data["linesearch_time"]
        iter_time_arr = np.pad(iter_time_arr, [(1, 0)])
        time_history_arr = np.cumsum(iter_time_arr)
        
        # construct dictionary benchmark_dict
        benchmark_dict = {}
        benchmark_dict['totalTime'] = summary_stats["total_time"]
        benchmark_dict['symbolic_fac_time'] = summary_stats["analyze_pattern_time"]
        benchmark_dict['numeric_fac_time'] = np.sum(table_data["factorization_time"]) 
        benchmark_dict['hessian_eval_time'] = np.sum(table_data["eval_hessian_time"]) + np.sum(table_data["eval_gradient_time"]) + np.sum(table_data["matrix_prep_time"])
        benchmark_dict['linear_solve_time'] = np.sum(table_data["solve_time"]) 
        return obj_arr, time_history_arr, grad_norm_arr, benchmark_dict

def derivativeEvalTiming(m, method, derivative_type, projection_type, repeat=10) -> float:
    '''
    m:                    the mesh read in MeshFEM
    method:               MeshFEM, TinyAD
    derivative_type:      AN(Analytical), FAD, TAD
    projection_type:      None, Fbased, Xbased
    
    return: per Eval timing
    '''
    
    # First filter out some invalid combinations
    if method == 'TinyAD' and projection_type == 'Fbased': raise RuntimeError('TinyAD method does not use Fbased projection!')
    if method == 'MeshFEM' and derivative_type == 'TAD' and projection_type == 'Fbased': raise RuntimeError('MeshFEM using TAD-style do not support Fbased projection!')
    
    # initial Tutte Embedding
    uv = mesh_energy.NodalVars(m, 2)
    bdry_uv = getBDdataOnNormalizedCircle(m)
    uv_init = tutteInitialization(m, bdry_uv)
    uv.setVars(uv_init.ravel())
    
    total_time = -100  # For error debug
    projectFlag = False
    if projection_type == 'None':  projectFlag = False
    elif projection_type in ['Fbased', 'Xbased']: projectFlag = True
    else: raise RuntimeError(f"projection_type: {projection_type} not valid in {method}!")
    
    if method == 'TinyAD':
        benchmark.reset()
        for i in range(repeat):
            f, g, H_proj = tinyad_parametrization.symmdsParamTinyADEvalFGH(m, uv_init.ravel(), project=projectFlag, proj_eps = 1e-9)
        bcmk_dict = benchmark.to_dict()
        total_time = benchmark.totalTime('symmdsParamTinyADEvalFGH', d=bcmk_dict)
    elif method == 'MeshFEM':
        if derivative_type == 'AN':
            symmdiri_energy = energy.SymmetricDirichlet(2)
            param = mesh_energy.Parametrization(m, uv, symmdiri_energy)
        elif derivative_type == 'FAD':
            symmdiri_energy = energy.SymmetricDirichletDerivativeFree(2)  
            param = mesh_energy.Parametrization(m, uv, symmdiri_energy)
        elif derivative_type == 'TAD':
            param = dirichlet_demo.param_symdirichlet_element_tad_compare(m, uv)
        else:  raise RuntimeError(f"derivative_type: {derivative_type} not valid in {method}!")
            
        if projection_type == 'Xbased':
            param.useXBasedProjection = True  # X-based
            param.xBasedProjectionClampEps = 1e-9
        elif projection_type == 'Fbased':
            param.useXBasedProjection = False
        
        p = py_newton_optimizer.NewtonMultiobjectiveProblem(uv, [param])
        p.disableCaching = True
        benchmark.reset()
        for i in range(repeat):
            p.gradient()
            p.hessian(projectFlag)
        bcmk_dict = benchmark.to_dict()
        total_time = benchmark.totalTime('NewtonMultiobjectiveProblem.hessian$', d=bcmk_dict) + benchmark.totalTime('NewtonMultiobjectiveProblem.gradient$', d=bcmk_dict)
        
    else: raise RuntimeError(f"method: {method} not supported.")
        
    return total_time/repeat

# Input Parameter:
# numCISBE: numConsecutiveIndefiniteStepsBeforeEnable
# numPSBD: numProjectionStepsBeforeDisable
def runSymmds_AdaptiveParameter(m, numCISBE, numPSBD, max_iter=200, hessian_shift=1e-5, grad_tol=None):
    import numpy as np
    obj_history = []
    time_history = []
    grad_norm_history = []
    hessian_projected_history = []
    hessian_shifted_amount_history = []
    step_norm_history = []
    directional_derivative_history = []

    def customCallback(prob, i):
        it_time = time.time()
        obj_history.append(prob.energy())
        time_history.append(it_time)
        grad_norm_history.append(np.linalg.norm(prob.gradient()))
        if i > 1:
            hessian_projected_history.append(int(prob.hessianWasProjected))
            hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)
    
    def customSaveStepDCallback(prob, step, directional_derivative):
        step_norm_history.append(np.linalg.norm(step))
        directional_derivative_history.append(-directional_derivative)

    uv = mesh_energy.NodalVars(m, 2)
    bdry_uv = getBDdataOnNormalizedCircle(m)
    uv_init = tutteInitialization(m, bdry_uv)
    uv.setVars(uv_init.ravel())

    symmdiri_energy = energy.SymmetricDirichlet(2)
    # Construct `SymmetricDirichlet` parametrization energy and problem
    param = mesh_energy.Parametrization(m, uv, symmdiri_energy)
    prob = py_newton_optimizer.NewtonMultiobjectiveProblem(uv, [param])
    prob.setCustomIterationCallback(customCallback)
    prob.setCustomLineSearchBeganCallback(customSaveStepDCallback)

    # Work around energy nullspace by adding a small shift
    prob.hessianShift = hessian_shift
    opt = prob.optimizer()
    opt.options.niter = max_iter
    opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAdaptive()
    opt.options.hessianProjectionController.numConsecutiveIndefiniteStepsBeforeEnable = numCISBE
    opt.options.hessianProjectionController.numProjectionStepsBeforeDisable = numPSBD
    if grad_tol is not None: opt.options.gradTol = grad_tol  # default is 2e-8

    benchmark.reset()
    start_time = time.time()
    cr = opt.optimize()
    # The projection status of the Hessian used in are i-1
    hessian_projected_history.append(int(prob.hessianWasProjected)) 
    hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)

    # Convert List to Array
    hessian_projected_arr = np.array(hessian_projected_history, dtype=int)
    hessian_shifted_arr = np.array(hessian_shifted_amount_history, dtype=float)
    hessian_indef_arr = np.array(cr.indefinite, dtype=int)
    step_size_arr = np.array(step_norm_history)
    dd_arr = np.array(directional_derivative_history)

    bk_dict = benchmark.to_dict()
    time_arr = np.array(time_history) - start_time

    return np.array(obj_history), time_arr, np.array(grad_norm_history), bk_dict, hessian_projected_arr, hessian_shifted_arr, hessian_indef_arr, step_size_arr, dd_arr



    


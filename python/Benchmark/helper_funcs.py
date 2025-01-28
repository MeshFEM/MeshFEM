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
import parametrization, py_newton_optimizer, benchmark
import tinyad_parametrization
import numpy as np
import copy, time
import igl

import numpy as np

def map_vertices_to_circle_area_normalized(V, F, bnd):
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
    BV = m.boundaryVertices()
    bloop = m.boundaryLoops()[0][::-1]
    bdry_uv = igl.map_vertices_to_circle(m.vertices(), BV[bloop])
    bdry_uv[bloop] =  bdry_uv.copy()
    return bdry_uv

def getBDdataOnNormalizedCircle(m):
    BV = m.boundaryVertices()
    bnd_loop = igl.boundary_loop(m.elements())
    bloop = np.searchsorted(BV, bnd_loop)
    bdry_uv = map_vertices_to_circle_area_normalized(m.vertices(), m.elements(), bnd_loop)
    bdry_uv[bloop] =  bdry_uv.copy()
    return bdry_uv

# read mesh and scale down vertices
def read_mesh(mesh_path : str):
    m_ori = mesh.Mesh(mesh_path)
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

def processEigenUVTXTs(folder_path):
    """
    Processes txt files in a given folder, converting them to raveled NumPy arrays
    and saving them as compressed .npz files. Deletes the original txt files after
    ensuring the same number of .npz files are created.

    Args:
        folder_path (str): Path to the folder containing the txt files.

    Returns:
        bool: True if the number of .npz files matches the original txt files, False otherwise.
    """
    # Get all files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.startswith("uv_Eigen_Iter_") and f.endswith(".txt")]
    npz_files_created = 0

    for txt_file in txt_files:
        try:
            # Build full path for the txt file
            txt_path = os.path.join(folder_path, txt_file)
            
            # Read the txt file into a NumPy array
            data = np.loadtxt(txt_path)
            
            # Ensure the data has two columns
            if data.ndim == 1 or data.shape[1] != 2:
                raise ValueError(f"File {txt_file} does not have two columns.")
            
            # Get the raveled version of the array
            raveled_data = data.ravel()
            
            # Construct the .npz file name
            file_index = txt_file.split('_')[-1].split('.')[0]  # Extract i from "uv_Eigen_Iter_i.txt"
            npz_file_name = f"uv_ravel_iter_{file_index}.npz"
            npz_path = os.path.join(folder_path, npz_file_name)
            
            # Save the raveled array to a compressed .npz file
            np.savez_compressed(npz_path, arr=raveled_data)
            npz_files_created += 1
        except Exception as e:
            print(f"Error processing file {txt_file}: {e}")

    # Check if the number of .npz files matches the number of txt files
    npz_files = [f for f in os.listdir(folder_path) if f.startswith("uv_ravel_iter_") and f.endswith(".npz")]
    if len(npz_files) == len(txt_files):
        # If numbers match, delete the txt files
        for txt_file in txt_files:
            try:
                os.remove(os.path.join(folder_path, txt_file))
            except Exception as e:
                print(f"Error deleting file {txt_file}: {e}")
        return True
    else:
        print(f"Mismatch in file counts: {len(txt_files)} txt files vs {len(npz_files)} npz files.")
        return False

def processUVTXTs(source_path, to_path, txt_file_prefix_str):
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
            file_index = txt_file.split('_')[-1].split('.')[0]  # Extract i from "uv_Eigen_Iter_i.txt"
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

def runSYDParam(m, max_iter=200, hessian_shift=1e-8, hessian_proj_option='Adaptive', grad_tol=None, uvsave_path=None):
    
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
    
    def customSaveUVCallback(prob, i):
        obj_history.append(prob.energy())
        grad_norm_history.append(np.linalg.norm(prob.gradient()))
        if i > 1:  hessian_projected_history.append(int(prob.hessianWasProjected))
        hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)
        uv_fn = 'uv_ravel_'+ 'iter_' + str(i-1)
        uv_arr = uv.getVars()
        np.savez_compressed(os.path.join(uvsave_path, uv_fn), arr=uv_arr)
    
    def customSaveStepDCallback(prob, step, directional_derivative):
        step_norm_history.append(np.linalg.norm(step))
        directional_derivative_history.append(-directional_derivative)

    uv = mesh_energy.NodalVars(m, 2)
    bdry_uv = getBDdataOnNormalizedCircle(m)
    uv_init = tutteInitialization(m, bdry_uv)
    uv.setVars(uv_init.ravel())

    if hessian_proj_option == 'AutoDiff':  symmdiri_energy = energy.SymmetricDirichletDerivativeFree(2)
    else:  symmdiri_energy = energy.SymmetricDirichlet(2)
    # Construct `SymmetricDirichlet` parametrization energy and problem
    param = mesh_energy.Parametrization(m, uv, symmdiri_energy)
    prob = py_newton_optimizer.NewtonMultiobjectiveProblem(uv, [param])
    if uvsave_path is None:  prob.setCustomIterationCallback(customCallback)
    else:
        if not os.path.exists(uvsave_path):
            raise RuntimeError(f"[Error] The uv_save path: {uvsave_path} does not exist!")
        prob.setCustomIterationCallback(customSaveUVCallback)
    prob.setCustomLineSearchBeganCallback(customSaveStepDCallback)

    # Work around energy nullspace by adding a small shift
    prob.hessianShift = hessian_shift
    opt = prob.optimizer()
    opt.options.niter = max_iter
    if hessian_proj_option == 'Adaptive' or hessian_proj_option == 'AutoDiff':  
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAdaptive()
    elif hessian_proj_option == 'Always':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()
    elif hessian_proj_option == 'Never':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionNever()
    elif hessian_proj_option == 'xbasedAlways':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()
        param.useXBasedProjection = True
    else:  raise RuntimeError("[Error] Usage of hessian_proj_option: Adaptive, Always, Never, xbasedAlways")
    if grad_tol is not None: opt.options.gradTol = grad_tol  # default is 2e-8

    benchmark.reset()
    start_time = time.time()
    opt.optimize()
    # benchmark.report()
    if uvsave_path is not None:      
        hessian_projected_history.append(int(prob.hessianWasProjected)) # The projection status of the Hessian used in are i-1
        hessian_shifted_amount_history.append(prob.lastFactorizationShiftMagnitude)

        obj_arr = np.array(obj_history)
        grad_norm_arr = np.array(grad_norm_history)
        # we saved uv coordinates per-iteration and hessian_projected_history
        hessian_projected_arr = np.array(hessian_projected_history, dtype=int)
        hessian_shifted_arr = np.array(hessian_shifted_amount_history, dtype=float)
        step_size_arr = np.array(step_norm_history)
        dd_arr = np.array(directional_derivative_history)

        obj_filename = 'obj_history.npy'
        grad_norm_filename = 'grad_norm_history.npy'
        hp_filename = 'hessian_projected_history.npy'
        hs_filename = 'hessian_shifted_amount_history.npy'
        step_filename = 'step_size_history.npy'
        dd_filename = 'directional_derivative_history.npy'

        np.save(os.path.join(uvsave_path, obj_filename), obj_arr)
        np.save(os.path.join(uvsave_path, grad_norm_filename), grad_norm_arr)
        np.save(os.path.join(uvsave_path, hp_filename), hessian_projected_arr)
        np.save(os.path.join(uvsave_path, hs_filename), hessian_shifted_arr)
        np.save(os.path.join(uvsave_path, step_filename), step_size_arr)
        np.save(os.path.join(uvsave_path, dd_filename), dd_arr)
        print(f"[File] Saved UV '.npz' files, {obj_filename}, {grad_norm_filename}, {hp_filename}, {hs_filename}, {step_filename}, {dd_filename} in {uvsave_path}.")
    else:
        bk_dict = benchmark.to_dict()
        time_arr = np.array(time_history) - start_time
        return np.array(obj_history), time_arr, np.array(grad_norm_history), bk_dict
    
def runSymmds_TinyAD(m, max_iter=200, grad_tol=2e-8, uvsave_path=None):

    bdry_uv = getBDdataOnNormalizedCircle(m)
    uv_init = tutteInitialization(m, bdry_uv)

    benchmark.reset()
    if uvsave_path is not None:
        uv_opt, obj_history, grad_history, time_history, step_size_history, dd_history = tinyad_parametrization.symmdsParamTinyAD(m, uv_init, max_iter, grad_tol, True, uvsave_path)
        # process all saved txt files into compressed npz files
        if not processEigenUVTXTs(uvsave_path):  raise RuntimeError(f"[Error] In Process Eigen txts in {uvsave_path}.")
        obj_arr = np.array(obj_history)
        grad_norm_arr = np.array(grad_history)
        step_size_arr = np.array(step_size_history)
        dd_arr = np.array(dd_history)

        obj_filename = 'obj_history.npy'
        grad_norm_filename = 'grad_norm_history.npy'
        step_filename = 'step_size_history.npy'
        dd_filename = 'directional_derivative_history.npy'

        np.save(os.path.join(uvsave_path, obj_filename), obj_arr)
        np.save(os.path.join(uvsave_path, grad_norm_filename), grad_norm_arr)
        np.save(os.path.join(uvsave_path, step_filename), step_size_arr)
        np.save(os.path.join(uvsave_path, dd_filename), dd_arr)
        print(f"[File] Saved UV '.npz' files, {obj_filename}, {grad_norm_filename}, {step_filename}, {dd_filename} in {uvsave_path}.")
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
        




    


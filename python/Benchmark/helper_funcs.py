'''
Python Helper Functions for benchmarking Parametrization using MeshFEM's new feature `MeshEnergy`

Author:  Xinzhuo (johnson) Hu
Created: 01/11/2025  2:07:55
'''

import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../')
import MeshFEM
import mesh, mesh_energy, energy
import parametrization, py_newton_optimizer, benchmark
import tinyad_parametrization
import numpy as np
import copy, time
import igl

def getBDdataOnUnitCircle(m):
    BV = m.boundaryVertices()
    bloop = m.boundaryLoops()[0][::-1]
    bdry_uv = igl.map_vertices_to_circle(m.vertices(), BV[bloop])
    bdry_uv[bloop] =  bdry_uv.copy()
    return bdry_uv

def tutteInitialization(m, bdry_uv):
    # Tutte Initialization
    uv_init = parametrization.harmonic(m, bdry_uv, False)
    flip_list = parametrization.getFlips(m, uv_init)
    if len(flip_list) > 0:  uv_init = parametrization.harmonic(m, bdry_uv, True)
    return uv_init

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

def runSYDParam(m, max_iter=200, hessian_shift=1e-8, hessian_proj_option='Adaptive', thread_num=0, grad_tol=None, 
                uvsave_path=None):
    
    os.environ['OMP_NUM_THREADS'] = '1'
    if thread_num > 0:
        import parallelism
        parallelism.set_max_num_tbb_threads = int(thread_num)

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
    bdry_uv = getBDdataOnUnitCircle(m)
    uv_init = tutteInitialization(m, bdry_uv)
    uv.setVars(uv_init.ravel())

    symmdiri_energy = energy.SymmetricDirichlet(2)
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
    if hessian_proj_option == 'Adaptive':  
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAdaptive()
    elif hessian_proj_option == 'Always':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()
    elif hessian_proj_option == 'Never':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionNever()
    elif hessian_proj_option == 'xbasedAlways':
        opt.options.hessianProjectionController = py_newton_optimizer.HessianProjectionAlways()
        param.useXBasedProjection = True
        prob.hessianShift = 0.0
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
    
def runSymmds_TinyAD(m, max_iter=100, thread_num=0, grad_tol=2e-8, uvsave_path=None):

    os.environ['OMP_NUM_THREADS'] = '1'
    if thread_num > 0:
        import parallelism
        parallelism.set_max_num_tbb_threads = int(thread_num)

    bdry_uv = getBDdataOnUnitCircle(m)
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



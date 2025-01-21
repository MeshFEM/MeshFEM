'''
Run Symmetric Dirichlet Parametrization for one model(mesh)

Author:  Xinzhuo (johnson) Hu
Created: 01/11/2025  11:03:50
'''

import os, sys
sys.path.append('../')
import MeshFEM, mesh, benchmark
import parallelism
import numpy as np
import pickle
import helper_funcs
import warnings

def saveStats(save_dir, obj_arr, time_arr, grad_norm_arr, benchmark_dict):
    if (obj_arr.shape[0] != time_arr.shape[0]):
        raise RuntimeWarning("[File] Array size mismatch of objective array and time array")
    if (obj_arr.shape[0] != grad_norm_arr.shape[0]):
        raise RuntimeWarning("[File] Array size mismatch of objective array and gradient norm array")
    arr_fn = 'obj_time_gradnorm.npz'
    dict_fn = 'benchmark_dict.pkl'
    np.savez_compressed(os.path.join(save_dir, arr_fn), obj_arr = obj_arr, time_arr = time_arr, grad_norm_arr=grad_norm_arr)
    # save benchmark dictionary
    with open(os.path.join(save_dir, dict_fn), "wb") as f:
        pickle.dump(benchmark_dict, f)
    
    print(f"[File] Successfully Write {arr_fn} and {dict_fn} in {save_dir}!")

def recordStatistics(base_path, model_name, model_path, hessian_proj_option, thread_num, repeat_num):
     # Print the parameters for confirmation
    print("-------------------------------------------------------------------------------------------------------------------")
    print(f"Running Symmetric Dirichelt Parametrization with the following parameters:")
    print(f"  Base Path: {base_path}")
    print(f"  Model Name: {model_name}")
    print(f"  Model Path: {model_path}")
    print(f"  Hessian Projection Option: {hessian_proj_option}")
    if thread_num == 0:  print(f"  Thread Number: Default")
    else:                print(f"  Thread Number: {thread_num}")
    print(f"  Repeat: {repeat_num}")

    # Statistics list
    total_time_list = []  # Used for searching the quickest experiment
    final_obj_list = []  
    newton_steps_list = []
    symbolic_factorize_time_list = []
    numeric_factroize_time_list = []
    linsys_solve_time_list = []
    hessian_eval_time_list = []
    line_search_time_list = []

    for i in range(repeat_num):
        print(f"Running parametrization experiment {i + 1}/{repeat_num}...")
        thread_folder_name = 'thread' + '_' + str(thread_num)
        folder_name = 'repeat' + '_' + str(i+1)
        folder_dir = os.path.join(base_path, model_name, hessian_proj_option, thread_folder_name, folder_name)
        if not os.path.exists(folder_dir):  os.makedirs(folder_dir)

        m = mesh.Mesh(model_path) # read mesh from model_path
        if hessian_proj_option == 'TinyAD':
            obj_arr, time_arr, grad_norm_arr, benchmark_dict = helper_funcs.runSymmds_TinyAD(m)
            line_search_time = benchmark.totalTime('Line Search$', d=benchmark_dict)
            linsys_solve_time = benchmark.totalTime('Linear Solve$', d=benchmark_dict)
            hessian_eval_time = benchmark.totalTime('Hessian Evaluation$', d=benchmark_dict)
            line_search_time_list.append(line_search_time)

        else:
            obj_arr, time_arr, grad_norm_arr, benchmark_dict = helper_funcs.runSYDParam(m, hessian_proj_option=hessian_proj_option)
            symbolic_factorize_time = benchmark.totalTime('Catamari Symbolic Factorize$', d=benchmark_dict)
            numeric_factorize_time = benchmark.totalTime('Catamari Numeric Factorize$', d=benchmark_dict)
            linsys_solve_time = benchmark.totalTime('CholeskyFactorizerBase.solve$', d=benchmark_dict)
            hessian_eval_time = benchmark.totalTime('NewtonMultiobjectiveProblem.hessian$', d=benchmark_dict)
            symbolic_factorize_time_list.append(symbolic_factorize_time)
            numeric_factroize_time_list.append(numeric_factorize_time)
        
        newton_steps = obj_arr.shape[0]
        total_time = time_arr[-1]
        final_obj = obj_arr[-1]

        total_time_list.append(total_time)
        final_obj_list.append(final_obj)
        newton_steps_list.append(newton_steps)
        linsys_solve_time_list.append(linsys_solve_time)
        hessian_eval_time_list.append(hessian_eval_time)

        print(f"[Opt] Symmetric Dirichlet Parametrization of {model_name} Ended in {newton_steps} Newton Steps. Total Elapsed Time: {total_time: .4f} seconds.")
        saveStats(folder_dir, obj_arr, time_arr, grad_norm_arr, benchmark_dict)

        print(f"Ended parametrization experiment {i + 1}/{repeat_num}.")
    
    outer_folder_dir = os.path.join(base_path, model_name, hessian_proj_option, thread_folder_name)
    # save to txt file
    txt_fn = 'summary.txt'
    if hessian_proj_option == 'TinyAD':
        summary_data = np.column_stack((newton_steps_list, total_time_list, final_obj_list, linsys_solve_time_list, hessian_eval_time_list, line_search_time_list))
        np.savetxt(os.path.join(outer_folder_dir, txt_fn), summary_data, fmt='%.8f', delimiter="\t", header="iter\t\ttime\t\tenergy\t\tlinsolve\thessian_eval\tline_search", comments='')

    else:
        summary_data = np.column_stack((newton_steps_list, total_time_list, final_obj_list, symbolic_factorize_time_list,
                                    numeric_factroize_time_list, linsys_solve_time_list, hessian_eval_time_list))
        np.savetxt(os.path.join(outer_folder_dir, txt_fn), summary_data, fmt='%.8f', delimiter="\t", header="iter\t\ttime\t\tenergy\t\tsymbol\t\tnumeric\t\tlinsolve\thessian_eval", comments='')
    
    # Find the fastest one and add it to 'summary.txt'
    total_time_min = min(total_time_list)
    index_min = total_time_list.index(total_time_min)
    with open(os.path.join(outer_folder_dir, txt_fn), "a") as f:
        f.write(f"Fastest: {index_min + 1}\n")

    print(f"[File] Successfully Write {txt_fn} in {outer_folder_dir}!")

def recordUV(base_path, model_name, model_path, hessian_proj_option):
    # Print the parameters for confirmation
    print("-------------------------------------------------------------------------------------------------------------------")
    print(f"Running Symmetric Dirichelt Parametrization Saving UV per-iteration with the following parameters:")
    print(f"  Base Path: {base_path}")
    print(f"  Model Name: {model_name}")
    print(f"  Model Path: {model_path}")
    print(f"  Hessian Projection Option: {hessian_proj_option}")

    save_uv_folder_name = 'UVs' # create a folder name 'UVs'
    folder_dir = os.path.join(base_path, model_name, hessian_proj_option, save_uv_folder_name)
    if not os.path.exists(folder_dir):  os.makedirs(folder_dir)

    m = mesh.Mesh(model_path)
    if hessian_proj_option == 'TinyAD':  helper_funcs.runSymmds_TinyAD(m, uvsave_path=folder_dir)
    else:                                helper_funcs.runSYDParam(m, hessian_proj_option=hessian_proj_option, uvsave_path=folder_dir)
    print(f"[File] Model: {model_name}. Hessian option: {hessian_proj_option} Saved UVs of all iterations in {folder_dir}.")

def main():
    # Ensure at least 5 arguments (excluding script name) are provided
    if len(sys.argv) < 6:
        print("Usage: python runSymmDiriParam.py <base_path> <model_name> <model_path> <hessian_proj_option> <save_uv_option> [<thread_num>] [<repeat_num>]")
        sys.exit(1)

    # Parse input arguments
    base_path = sys.argv[1]
    model_name = sys.argv[2]
    model_path = sys.argv[3]
    hessian_proj_option = sys.argv[4]
    save_uv_option = sys.argv[5]

    if (hessian_proj_option not in ['Adaptive', 'Always', 'Never', 'xbasedAlways', 'TinyAD']):
        print("[Error] Usage of <hessian_proj_option>:  Adaptive or Always or Never or xbasedAlways or TinyAD")
        sys.exit(1)
    
    if (save_uv_option.lower() not in ['yes', 'no', 'both']):
        print("[Error] Usage of <save_uv_option>:  yes or no or both")
        sys.exit(1)

    # Set default value for thread_num iter_num if not provided
    thread_num = int(sys.argv[6]) if len(sys.argv) > 6 else 0  # thread_num is 0 means using default thread number
    repeat_num = int(sys.argv[7]) if len(sys.argv) > 7 else 1

    # Check if base_path exists
    if not os.path.exists(base_path):
        warnings.warn(f"Warning: The base_path '{base_path}' does not exist. Please check the path.")
        sys.exit(1)  # Exit if the path does not exist
    
    if hessian_proj_option == 'TinyAD':
        if thread_num != 0: # not in default case
            os.environ['OMP_NUM_THREADS'] = str(thread_num)
    else:  
        os.environ['OMP_NUM_THREADS'] = '1'
        parallelism.set_max_num_tbb_threads(int(thread_num))
    
    if save_uv_option.lower() == 'no':
        recordStatistics(base_path, model_name, model_path, hessian_proj_option, thread_num, repeat_num)
    
    if save_uv_option.lower() == 'yes':
        recordUV(base_path, model_name, model_path, hessian_proj_option)
    
    if save_uv_option.lower() == 'both':
        recordStatistics(base_path, model_name, model_path, hessian_proj_option, thread_num, repeat_num)
        recordUV(base_path, model_name, model_path, hessian_proj_option)

if __name__ == "__main__":
    main()

'''
Run Symmetric Dirichlet Parametrization for one model(mesh)

Author:  Xinzhuo (johnson) Hu
Created: 01/11/2025  11:03:50
'''

import os, sys
sys.path.append('../')
import MeshFEM, mesh
import numpy as np
import pickle
import helper_funcs
import warnings

def saveStats(save_dir, obj_arr, time_arr, benchmark_dict):
    if (obj_arr.shape[0] != time_arr.shape[0]):
        raise RuntimeWarning("[File] Array size mismatch of objective array and time array")
    arr_fn = 'obj_and_time.npz'
    dict_fn = 'benchmark_dict.pkl'
    np.savez_compressed(os.path.join(save_dir, arr_fn), obj_arr = obj_arr, time_arr = time_arr)
    # save benchmark dictionary
    with open(os.path.join(save_dir, dict_fn), "wb") as f:
        pickle.dump(benchmark_dict, f)
    
    print(f"[File] Successfully Write {arr_fn}('obj_arr' and 'time_arr') and {dict_fn} in {save_dir}!")

def main():
    # Ensure at least 4 arguments (excluding script name) are provided
    if len(sys.argv) < 5:
        print("Usage: python runSymmDiriParam.py <base_path> <model_name> <model_path> <hessian_proj_option> [<repeat_num>]")
        sys.exit(1)

    # Parse input arguments
    base_path = sys.argv[1]
    model_name = sys.argv[2]
    model_path = sys.argv[3]
    hessian_proj_option = sys.argv[4]

    if (hessian_proj_option != 'Adaptive') and (hessian_proj_option != 'Always') and (hessian_proj_option != 'Never'):
        print("[Error] Usage of <hessian_proj_option>:  Adaptive or Always or Never")
        sys.exit(1)

    # Set default value for iter_num if not provided
    repeat_num = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    # Check if base_path exists
    if not os.path.exists(base_path):
        warnings.warn(f"Warning: The base_path '{base_path}' does not exist. Please check the path.")
        sys.exit(1)  # Exit if the path does not exist

    # Print the parameters for confirmation
    print(f"Running Symmetric Dirichelt Parametrization with the following parameters:")
    print(f"  Base Path: {base_path}")
    print(f"  Model Name: {model_name}")
    print(f"  Model Path: {model_path}")
    print(f"  Hessian Projection Option: {hessian_proj_option}")
    print(f"  Iterations: {repeat_num}")

    total_time_list = []  # Used for searching the quickest experiment
    final_obj_list = []  
    newton_steps_list = []

    for i in range(repeat_num):
        print(f"Running parametrization experiment {i + 1}/{repeat_num}...")
        folder_name = 'repeat' + '_' + str(i+1)
        folder_dir = os.path.join(base_path, model_name, hessian_proj_option, folder_name)
        if not os.path.exists(folder_dir):  os.makedirs(folder_dir)

        m = mesh.Mesh(model_path)
        obj_arr, time_arr, benchmark_dict = helper_funcs.runSYDParam(m, hessian_proj_option=hessian_proj_option)
        
        newton_steps = obj_arr.shape[0]
        total_time = time_arr[-1]
        final_obj = obj_arr[-1]
        total_time_list.append(total_time)
        final_obj_list.append(final_obj)
        newton_steps_list.append(newton_steps)

        print(f"[Opt] Symmetric Dirichlet Parametrization of {model_name} Ended in {newton_steps} Newton Steps. Total Elapsed Time: {total_time: .4f} seconds.")
        saveStats(folder_dir, obj_arr, time_arr, benchmark_dict)

        print(f"Ended parametrization experiment {i + 1}/{repeat_num}.")
    
    outer_folder_dir = os.path.join(base_path, model_name, hessian_proj_option)
    summary_data = np.column_stack((newton_steps_list, total_time_list, final_obj_list))
    # save to txt file
    txt_fn = 'summary.txt'
    np.savetxt(os.path.join(outer_folder_dir, txt_fn), summary_data, fmt='%.8f', delimiter="\t", header="iter\t\ttime\t\tenergy", comments='')
    print(f"[File] Successfully Write {txt_fn} in {outer_folder_dir}!")


if __name__ == "__main__":
    main()

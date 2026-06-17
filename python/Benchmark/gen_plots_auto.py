'''
Auto scripts to generate figures for each model

Author:  Xinzhuo (johnson) Hu
Created: 01/14/2025  10:07:12pm
'''
import os, sys
sys.path.append('../')
import MeshFEM, mesh, benchmark
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

def getFastestRepeatIndex(directory):
    file_path = os.path.join(directory, "summary.txt")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"'summary.txt' not found in {directory}")
    # Read the last line of the file
    with open(file_path, "r") as file:
        lines = file.readlines()
        if not lines:  raise RuntimeError(f"'summary.txt' is empty in {directory}")
        return (lines[-1].strip())[-1]

def read_benchmark_data(directory):
    # Full paths to the files
    pkl_file_path = os.path.join(directory, "benchmark_dict.pkl")
    npz_file_path = os.path.join(directory, "obj_time_gradnorm.npz")
    
    # Check if files exist
    if not os.path.isfile(pkl_file_path):
        raise FileNotFoundError(f"'benchmark_dict.pkl' not found in {directory}")
    if not os.path.isfile(npz_file_path):
        raise FileNotFoundError(f"'obj_and_time.npz' not found in {directory}")

    # Load the dictionary from pickle file
    with open(pkl_file_path, "rb") as f:
        benchmark_dict = pickle.load(f)

    # Load numpy arrays from .npz file
    npz_data = np.load(npz_file_path)
    obj_arr = npz_data['obj_arr']
    time_arr = npz_data['time_arr']
    grad_norm_arr = npz_data['grad_norm_arr']
    return obj_arr, time_arr, grad_norm_arr, benchmark_dict

# Plot different hessian projection options under one thread configuration
def save_obj_grad_time_figure(obj_grad_time_list, thread_num, model_name, directory):
    tn = thread_num
    iterations_list = []  # iteration numbers for each hessian projection option
    for i in range(3):
        iterations = np.arange(0, obj_grad_time_list[i][tn].shape[1])
        iterations_list.append(iterations)
    
    hessian_option_list = ['Adaptive', 'Always', 'Never']
    color_list = ['dodgerblue', 'magenta', 'tomato']
    line_style_list = ['-', '--', '-.']
    
    # Generate plt
    plt.figure(figsize=(12, 12))
    plt.subplot(2,2,1)
    for i in range(3):
        plt.plot(iterations_list[i], obj_grad_time_list[i][tn][0], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
    plt.title(f"Model: {model_name}", fontsize=16)
    plt.yscale('log')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(" Energy ", fontsize=14)
    plt.legend()
    
    plt.subplot(2,2,2)
    for i in range(3):
        plt.plot(iterations_list[i], obj_grad_time_list[i][tn][1], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
    plt.title(f"Model: {model_name}", fontsize=16)
    plt.yscale('log')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(" Grad Norm ", fontsize=14)
    plt.legend()

    plt.subplot(2,2,3)
    for i in range(3):
        plt.plot(obj_grad_time_list[i][tn][2], obj_grad_time_list[i][tn][0], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])

    plt.yscale('log')
    plt.xlabel("Time [sec]", fontsize=12)
    plt.ylabel(" Energy ", fontsize=14)
    plt.legend()

    plt.subplot(2,2,4)
    for i in range(3):
        plt.plot(obj_grad_time_list[i][tn][2], obj_grad_time_list[i][tn][1], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
    plt.yscale('log')
    plt.xlabel("Time [sec]", fontsize=12)
    plt.ylabel("Grad Norm", fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    full_fn = model_name + '_objgradvsT' + '_thread' + str(thread_num) + '.png'
    plt.savefig(os.path.join(directory, full_fn), dpi=300)
    print(f"[Plot] '{full_fn}' saved in {directory}!")


def gen_plots(base_path, thread_num_list):
    model_name_list = [entry.name for entry in os.scandir(base_path) if entry.is_dir()]
    hessian_option_list = ['Adaptive', 'Always', 'Never']

    total_timer = time.time()
    for model_name in model_name_list:
        obj_grad_time_list = [[], [], []]
        for hessopt_ind, hessian_option in enumerate(hessian_option_list):
            for thread_num in thread_num_list:
                thread_dir_name = 'thread' + '_' + str(thread_num)
                cur_dir = os.path.join(base_path, model_name, hessian_option, thread_dir_name)
                # read file 'summary.txt'
                fast_ind = getFastestRepeatIndex(cur_dir)
                if fast_ind == '0':  fast_ind = '10'
                repeat_dir_name = 'repeat' + '_' + fast_ind
                data_dir = os.path.join(cur_dir, repeat_dir_name)
                obj_arr, time_arr, grad_norm_arr, benchmark_dict = read_benchmark_data(data_dir)
                obj_grad_time = np.vstack((obj_arr, grad_norm_arr, time_arr)) # make a (3,n) numpy array
                obj_grad_time_list[hessopt_ind].append(obj_grad_time)
        
        plot_path = os.path.join(base_path, model_name)
        for thread_num in thread_num_list:
            save_obj_grad_time_figure(obj_grad_time_list, thread_num, model_name, plot_path)
    
    total_time = time.time() - total_timer
    print(f"All plots generated! Time: {total_time:.4f} seconds.")
    print("---------------------------------------------------------------------------------------")


if __name__ == "__main__":
    # Check if the script is provided with the required arguments
    if len(sys.argv) != 2:
        print("Usage: python gen_plots_auto.py <result_path>")
        sys.exit(1)

    # Parse command-line arguments
    result_path = sys.argv[1]
    
    thread_num_list = [0]
    # thread_num_list = [1, 2, 4, 8, 16]
    # Run the function to perform experiments
    gen_plots(result_path, thread_num_list)
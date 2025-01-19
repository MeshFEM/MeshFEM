'''
Python Helper Functions for reading experiment data (obj, gradnorm, time, UVs...)
and creating plots and videos

Author:  Xinzhuo (johnson) Hu
Created: 01/16/2025  19:42:55
'''

import os, sys
sys.path.append('../')
import MeshFEM, mesh, mesh_energy,viewer, benchmark
import mesh_operations
import numpy as np
import math, bisect, time
import pickle
import video_writer
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

def getColorLineList(options):
    color_list = ['dodgerblue', 'magenta', 'tomato']
    line_style_list = ['-', '--', '-.']
    if options == 4:
        color_list.append('forestgreen')
        line_style_list.append('-')
    return color_list, line_style_list

def create_list_of_lists(num_inner_lists):
    """
    Create a list containing a specified number of empty inner lists.

    Args:
        num_inner_lists (int): The number of inner lists to create.

    Returns:
        list: A list containing `num_inner_lists` empty inner lists.
    """
    return [[] for _ in range(num_inner_lists)]

# Under Hessian_Option/UVs
# The directory should contain 'uv_ravel_iter_i.npz' data
def getNumUVs(directory):  return sum(1 for f in os.listdir(directory) if f.endswith(".npz"))

def read_uv_data(directory, i):
    uv_fn = 'uv_ravel_iter_' + str(i) + '.npz'
    uv_data = np.load(os.path.join(directory, uv_fn))
    uv_ravel = uv_data['arr']
    return uv_ravel

def read_uv_min_data(directory):
    largest_i = getNumUVs(directory) - 1
    uv_fn = 'uv_ravel_iter_' + str(largest_i) + '.npz'
    uv_data = np.load(os.path.join(directory, uv_fn))
    uv_min = uv_data['arr']
    return uv_min

def compute_uv_distance(directory):
    numUVs = getNumUVs(directory)
    uv_min = read_uv_min_data(directory)
    
    dist_list = []
    for i in range(numUVs-1):
        uv_fn = 'uv_ravel_iter_' + str(i) + '.npz'
        uv_npz = np.load(os.path.join(directory, uv_fn))
        uv_data = uv_npz['arr']
        dist = np.linalg.norm(uv_data - uv_min)
        dist /= np.linalg.norm(np.max(uv_min, axis=0) - np.min(uv_min, axis=0))
        dist_list.append(dist)
    
    return dist_list

def readOptionData(directory, filename):
    filepath = os.path.join(directory, filename)
    option_data = np.array([])
    if os.path.exists(filepath):  option_data = np.load(filepath)
    if option_data.size == 0:
        print(f"File '{filename}' not found in {directory}. Initialized as an empty NumPy array.")
    return option_data


def read_HessianProjected_data(directory):
    obj_filename = 'obj_history.npy'
    grad_norm_filename = 'grad_norm_history.npy'
    obj_data = np.load(os.path.join(directory, obj_filename))
    grad_norm_data = np.load(os.path.join(directory, grad_norm_filename))

    hp_file_name = 'hessian_projected_history.npy'
    hs_file_name = 'hessian_shifted_amount_history.npy'
    hessian_projected_data = readOptionData(directory, hp_file_name)
    hessian_shifted_amount_data = readOptionData(directory, hs_file_name)

    return obj_data, grad_norm_data, hessian_projected_data, hessian_shifted_amount_data

# Under Hessian_Option/thread_i/
def getFastestRepeatIndex(directory):
    # We assume the 0 < repeat number <=10
    file_path = os.path.join(directory, "summary.txt")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"'summary.txt' not found in {directory}")
    # Read the last line of the file
    with open(file_path, "r") as file:
        lines = file.readlines()
        if not lines:  raise RuntimeError(f"'summary.txt' is empty in {directory}")
        fast_ind = (lines[-1].strip())[-1]
        if fast_ind == '0': fast_ind = '10'
        return fast_ind

# Under Hessian_Option/thread_i/repeat_i
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

# For all Hessian_Option/
# Input parameter: directory = base_path/user_model_name
def readConvergenceTimingData(directory, thread_num_list=[0], hessian_option_list = ['Adaptive', 'Always', 'Never']):
    obj_grad_time_list = create_list_of_lists(len(hessian_option_list))
    for hessopt_ind, hessian_option in enumerate(hessian_option_list):
        # read obj, grad_norm, timing data
        for thread_num in thread_num_list:
            thread_dir_name = 'thread' + '_' + str(thread_num)
            cur_dir = os.path.join(directory, hessian_option, thread_dir_name)
            fast_ind = getFastestRepeatIndex(cur_dir) # read file 'summary.txt'
            repeat_dir_name = 'repeat' + '_' + fast_ind
            data_dir = os.path.join(cur_dir, repeat_dir_name)
            obj_arr, time_arr, grad_norm_arr, benchmark_dict = read_benchmark_data(data_dir)
            obj_grad_time = np.vstack((obj_arr, grad_norm_arr, time_arr)) # make a (3,n) numpy array
            obj_grad_time_list[hessopt_ind].append(obj_grad_time)
    
    return obj_grad_time_list

# For all Hessian_Option/
# Input parameter: directory = base_path/user_model_name
def readHessianData(directory, hessian_option_list = ['Adaptive', 'Always', 'Never']):
    obj_list = []
    grad_norm_list = []
    hessian_projected_list = []
    hessian_shifted_amount_list = []

    for hessopt_ind, hessian_option in enumerate(hessian_option_list):
        
        # read UV_related Hessian data
        uv_folder_name = 'UVs'
        uv_dir = os.path.join(directory, hessian_option, uv_folder_name)
        obj_data, grad_norm_data, hessian_projected_data, hessian_shifted_amount_data = read_HessianProjected_data(uv_dir)

        obj_list.append(obj_data)
        grad_norm_list.append(grad_norm_data)
        hessian_projected_list.append(hessian_projected_data)
        hessian_shifted_amount_list.append(hessian_shifted_amount_data)
    
    return obj_list, grad_norm_list, hessian_projected_list, hessian_shifted_amount_list

# For all Hessian_Option/
# Input parameter: directory = base_path/user_model_name
def readUVdist(directory, hessian_option_list = ['Adaptive', 'Always', 'Never']):
    uv_dist_list = []
    for hessopt_ind, hessian_option in enumerate(hessian_option_list):
        # compute uv distance
        uv_folder_name = 'UVs'
        uv_dir = os.path.join(directory, hessian_option, uv_folder_name)
        uv_dist_arr = np.array(compute_uv_distance(uv_dir))
        uv_dist_list.append(uv_dist_arr)
    return uv_dist_list

# align timing
def alignTiming(obj_grad_time_list, grad_norm_list, thread_ind=0):
    aligned_timing_list = []
    for i in range(len(grad_norm_list)):
        timing_list = obj_grad_time_list[i][thread_ind][-1].copy()
        num_time_steps = timing_list.shape[0]
        num_grad_steps = grad_norm_list[i].shape[0]
        if num_time_steps == num_grad_steps:
            aligned_timing_list.append(timing_list)
        elif num_time_steps > num_grad_steps:
            sliced_timing_list = timing_list[:num_grad_steps]
            aligned_timing_list.append(sliced_timing_list)
        else:
            diff_steps = num_grad_steps - num_time_steps
            iter_time_list = []
            for i in range(diff_steps):
                ind = -1 - i
                iter_time = (timing_list[ind] - timing_list[ind-1])
                iter_time_list.append(iter_time)
            reverse_iter_time_list = iter_time_list[::-1]
            append_time_list = []
            append_time_counter = timing_list[-1]
            for iter_time in reverse_iter_time_list:
                append_time_counter += iter_time
                append_time_list.append(append_time_counter)
            aligned_timing_list.append(np.array(timing_list.tolist() + append_time_list))
        
    return aligned_timing_list

def getStepsMaxMin_FromMetricList(metric_list):
    num_options = len(metric_list)
    max_steps = 0
    max_metric = 0
    min_metric = float('inf')
    for i in range(num_options):
        if metric_list[i].shape[0] > max_steps:  max_steps = metric_list[i].shape[0]
        if np.max(metric_list[i]) > max_metric: max_metric = np.max(metric_list[i])
        if np.min(metric_list[i] < min_metric): min_metric = np.min(metric_list[i])
    return max_steps, max_metric, min_metric


# Plot different hessian projection options under one thread configuration
def save_obj_grad_time_figure(obj_grad_time_list, user_model_name, save_directory, thread_ind=0, 
                              thread_num_list=[0], hessian_option_list=['Adaptive', 'Always', 'Never']):
    tn = thread_ind
    num_options = len(hessian_option_list)
    iterations_list = []  # iteration numbers for each hessian projection option
    for i in range(num_options):
        iterations = np.arange(0, obj_grad_time_list[i][tn].shape[1])
        iterations_list.append(iterations)
    
    color_list, line_style_list = getColorLineList(num_options)
    
    # Generate plt
    plt.figure(figsize=(12, 12))
    plt.subplot(2,2,1)
    for i in range(num_options):
        plt.plot(iterations_list[i], obj_grad_time_list[i][tn][0], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
    plt.title(f"Model: {user_model_name}", fontsize=16)
    plt.yscale('log')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(" Energy ", fontsize=14)
    plt.legend()
    
    plt.subplot(2,2,2)
    for i in range(num_options):
        plt.plot(iterations_list[i], obj_grad_time_list[i][tn][1], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
    plt.title(f"Model: {user_model_name}", fontsize=16)
    plt.yscale('log')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(" Grad Norm ", fontsize=14)
    plt.legend()

    plt.subplot(2,2,3)
    for i in range(num_options):
        plt.plot(obj_grad_time_list[i][tn][2], obj_grad_time_list[i][tn][0], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])

    plt.yscale('log')
    plt.xlabel("Time [sec]", fontsize=12)
    plt.ylabel(" Energy ", fontsize=14)
    plt.legend()

    plt.subplot(2,2,4)
    for i in range(num_options):
        plt.plot(obj_grad_time_list[i][tn][2], obj_grad_time_list[i][tn][1], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
    plt.yscale('log')
    plt.xlabel("Time [sec]", fontsize=12)
    plt.ylabel("Grad Norm", fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    full_fn = user_model_name + '_objgradvsT' + '_thread' + str(thread_num_list[thread_ind]) + '.png'
    plt.savefig(os.path.join(save_directory, full_fn), dpi=300)
    print(f"[Plot] '{full_fn}' saved in {save_directory}!")
    plt.close()

# Plot GradNorm vs iter 
def save_grad_iter_figure(grad_norm_list, hessian_projected_list, user_model_name, save_directory, hessian_option_list = ['Adaptive', 'Always', 'Never']):
    iterations_list = []
    num_options = len(hessian_option_list)
    for i in range(num_options):
        iterations = np.arange(0, grad_norm_list[i].shape[0])
        iterations_list.append(iterations)
    
    color_list, line_style_list = getColorLineList(num_options)

    adaptive_projtrue_iter = iterations_list[0][hessian_projected_list[0]==1]
    grad_norm_projtrue_list = grad_norm_list[0][hessian_projected_list[0]==1]

    plt.figure(figsize=(8, 8))
    for i in range(num_options):
        plt.plot(iterations_list[i], grad_norm_list[i], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
    plt.scatter(adaptive_projtrue_iter, grad_norm_projtrue_list, color='blue', marker='o', s=80)
    plt.title(f"Model: {user_model_name}", fontsize=16)
    plt.yscale('log')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(" Grad Norm ", fontsize=14)
    plt.legend()
    plt.tight_layout()

    plot_name = user_model_name + '_GradIterScatter.png'
    plt.savefig(os.path.join(save_directory, plot_name), dpi=300)
    print(f"[Plot] '{plot_name}' saved in {save_directory}!")
    plt.close()

# Plot Obj vs iter 
def save_obj_iter_figure(obj_list, hessian_projected_list, user_model_name, save_directory, hessian_option_list = ['Adaptive', 'Always', 'Never']):
    iterations_list = []
    num_options = len(hessian_option_list)
    for i in range(num_options):
        iterations = np.arange(0, obj_list[i].shape[0])
        iterations_list.append(iterations)
    
    color_list, line_style_list = getColorLineList(num_options)

    adaptive_projtrue_iter = iterations_list[0][hessian_projected_list[0]==1]
    grad_norm_projtrue_list = obj_list[0][hessian_projected_list[0]==1]

    plt.figure(figsize=(8, 8))
    for i in range(num_options):
        plt.plot(iterations_list[i], obj_list[i], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
    plt.scatter(adaptive_projtrue_iter, grad_norm_projtrue_list, color='blue', marker='o', s=80)
    plt.title(f"Model: {user_model_name}", fontsize=16)
    plt.yscale('log')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(" Energy ", fontsize=14)
    plt.legend()
    plt.tight_layout()

    plot_name = user_model_name + '_ObjIterScatter.png'
    plt.savefig(os.path.join(save_directory, plot_name), dpi=300)
    print(f"[Plot] '{plot_name}' saved in {save_directory}!")
    plt.close()

# Plot distance to final converged UV configuration
# Under different hessian projection options under one thread configuration
def save_uv_dist_figure(uv_dist_list, hessian_projected_list, user_model_name, save_directory, hessian_option_list = ['Adaptive', 'Always', 'Never']):
    num_options = len(hessian_option_list)
    iterations_list = []
    for i in range(num_options):
        iterations = np.arange(0, len(uv_dist_list[i]))
        iterations_list.append(iterations)

    color_list, line_style_list = getColorLineList(num_options)

    modified_hessian_proj_list = hessian_projected_list[0][:-1] # because in uv distance plot we ignore the last step
    adaptive_projtrue_iter = iterations_list[0][modified_hessian_proj_list==1]
    uv_dist_projtrue_list = uv_dist_list[0][modified_hessian_proj_list==1]
    
    plt.figure(figsize=(8, 8))
    for i in range(num_options):
        plt.plot(iterations_list[i], uv_dist_list[i], ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
    
    plt.scatter(adaptive_projtrue_iter, uv_dist_projtrue_list, color='blue', marker='o', s=30)
    plt.title(f"Model: {user_model_name}", fontsize=16)
    plt.yscale('log')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(" Distance to Minimum ", fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    full_fn = user_model_name + '_UVdistScatter.png'
    plt.savefig(os.path.join(save_directory, full_fn), dpi=300)
    print(f"[Plot] '{full_fn}' saved in {save_directory}!")
    plt.close()

# generate plot videos showing the grad norm vs iterations plots under 3 different Hessian options
# grad_norm_list, hessian_projected_list: readHessianData(...)
# obj_grad_time_list: readConvergenceTimingData(...)
# fps and figsize
def gen_GradIter_videos(grad_norm_list, hessian_projected_list, obj_grad_time_list, user_model_name, 
                        save_directory, thread_ind = 0, thread_num_list=[0], hessian_option_list=['Adaptive', 'Always', 'Never'],
                        fps=30, default_fig_size=(8, 8)):
    
    num_options = len(hessian_option_list)
    max_num_steps, max_grad_norm, min_grad_norm = getStepsMaxMin_FromMetricList(grad_norm_list)

    max_N_grad_norm = math.ceil(math.log10(max_grad_norm)) + 1
    min_N_grad_norm = math.floor(math.log10(min_grad_norm)) - 1

    grad_iter_vdname = user_model_name + '_GradVSIter' + '_thread' + str(thread_num_list[thread_ind]) + '.mp4'
    fig = plt.figure(figsize=default_fig_size)

    plt.xlim(0, max_num_steps)
    plt.ylim(10**(min_N_grad_norm), 10**(max_N_grad_norm))
    plt.title(f"Model: {user_model_name}", fontsize=16)
    plt.yscale('log')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(" Grad Norm ", fontsize=14)
    plt.legend()
    pw = video_writer.PlotVideoWriter(os.path.join(save_directory, grad_iter_vdname), plt.gcf(), dpi=300, )

    iterations_list = []
    for i in range(num_options):
        iterations = np.arange(0, grad_norm_list[i].shape[0])
        iterations_list.append(iterations)
    color_list, line_style_list = getColorLineList(num_options)
    
    spf = 1 / fps
    aligned_timing_list = alignTiming(obj_grad_time_list, grad_norm_list, thread_ind=thread_ind)
    _, totalTime, _ = getStepsMaxMin_FromMetricList(aligned_timing_list)
    numFrames = int(math.ceil(totalTime / spf))

    adaptive_projtrue_iter = iterations_list[0][hessian_projected_list[0]==1]
    grad_norm_projtrue_list = grad_norm_list[0][hessian_projected_list[0]==1]
    start_record_timer = time.time()
    for f in range(numFrames):
        frameTime = f * spf
        fig = plt.figure(figsize=default_fig_size)
        index_for_dots = 0
        for i in range(num_options):
            iterationForFrame = max(0, bisect.bisect_right(aligned_timing_list[i], frameTime) - 1)
            plt.plot(iterations_list[i][:iterationForFrame+1], grad_norm_list[i][:iterationForFrame+1], 
                    ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
            if i == 0: index_for_dots = max(0, bisect.bisect_left(adaptive_projtrue_iter, iterationForFrame))
            plt.scatter(adaptive_projtrue_iter[:index_for_dots], grad_norm_projtrue_list[:index_for_dots], color='blue', marker='o', s=60)
        
        plt.xlim(0, max_num_steps)
        plt.ylim(10**(min_N_grad_norm), 10**(max_N_grad_norm))

        plt.title(f"Model: {user_model_name}", fontsize=16)
        plt.yscale('log')
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel(" Grad Norm ", fontsize=14)
        plt.legend(loc="upper right")
        pw.writeFrame(plt.gcf())
        plt.close()
    pw.finish()
    elapsed_record_time = time.time() - start_record_timer
    print(f"[Viedo] {grad_iter_vdname} recorded in {save_directory}. Time: {elapsed_record_time:.4f} seconds.")

# generate plot videos showing the objective(energy) vs iterations plots under 3 different Hessian options
# grad_norm_list, hessian_projected_list: readHessianData(...)
# obj_grad_time_list: readConvergenceTimingData(...)
# fps and figsize
def gen_ObjIter_videos(obj_list, hessian_projected_list, obj_grad_time_list, user_model_name, 
                        save_directory, thread_ind = 0, thread_num_list=[0], hessian_option_list=['Adaptive', 'Always', 'Never'],
                        fps=30, default_fig_size=(8, 8)):
    
    num_options = len(hessian_option_list)
    max_num_steps, max_obj, min_obj = getStepsMaxMin_FromMetricList(obj_list)
    max_N_obj = math.ceil(math.log10(max_obj)) + 1
    min_N_obj = math.floor(math.log10(min_obj)) - 1

    obj_iter_vdname = user_model_name + '_ObjVSIter' + '_thread' + str(thread_num_list[thread_ind]) + '.mp4'
    fig = plt.figure(figsize=default_fig_size)
    plt.xlim(0, max_num_steps)
    plt.ylim(10**(min_N_obj), 10**(max_N_obj))
    plt.title(f"Model: {user_model_name}", fontsize=16)
    plt.yscale('log')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel(" Energy ", fontsize=14)
    plt.legend()
    pw = video_writer.PlotVideoWriter(os.path.join(save_directory, obj_iter_vdname), plt.gcf(), dpi=300, )

    iterations_list = []
    for i in range(num_options):
        iterations = np.arange(0, obj_list[i].shape[0])
        iterations_list.append(iterations)
    color_list, line_style_list = getColorLineList(num_options)

    spf = 1 / fps
    aligned_timing_list = alignTiming(obj_grad_time_list, obj_list, thread_ind=thread_ind)
    _, totalTime, _ = getStepsMaxMin_FromMetricList(aligned_timing_list)
    numFrames = int(math.ceil(totalTime / spf))

    adaptive_projtrue_iter = iterations_list[0][hessian_projected_list[0]==1]
    obj_projtrue_list = obj_list[0][hessian_projected_list[0]==1]

    start_record_timer = time.time()
    for f in range(numFrames):
        frameTime = f * spf
        fig = plt.figure(figsize=default_fig_size)
        for i in range(num_options):
            iterationForFrame = max(0, bisect.bisect_right(aligned_timing_list[i], frameTime) - 1)
            plt.plot(iterations_list[i][:iterationForFrame+1], obj_list[i][:iterationForFrame+1], 
                    ls=line_style_list[i], color=color_list[i], label=hessian_option_list[i])
            if i == 0: index_for_dots = max(0, bisect.bisect_left(adaptive_projtrue_iter, iterationForFrame))
            plt.scatter(adaptive_projtrue_iter[:index_for_dots], obj_projtrue_list[:index_for_dots], color='blue', marker='o', s=60)
        
        plt.xlim(0, max_num_steps)
        plt.ylim(10**(min_N_obj), 10**(max_N_obj))

        plt.title(f"Model: {user_model_name}", fontsize=16)
        plt.yscale('log')
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel(" Energy ", fontsize=14)
        plt.legend(loc="upper right")
        pw.writeFrame(plt.gcf()) 
        plt.close()
    pw.finish()
    elapsed_record_time = time.time() - start_record_timer
    print(f"[Viedo] {obj_iter_vdname} recorded in {save_directory}. Time: {elapsed_record_time:.4f} seconds.")

# Generate videos recording parametrization process of models under 3 different Hessian Options and one thread
# metric_list: any list return from readHessianData(...)
# obj_grad_time_list: acquire timing list for specific thread
def gen_Param_videos(base_path, metric_list, obj_grad_time_list, user_model_name, model_base_path, save_directory, 
                     thread_ind=0, thread_num_list=[0], hessian_option_list=['Adaptive', 'Always', 'Never'], fps=30):
    
    num_options = len(hessian_option_list)
    model_name_fex = user_model_name + '.off'
    model_path = os.path.join(model_base_path, model_name_fex)
    m = mesh.Mesh(model_path)

    aligned_timing_list = alignTiming(obj_grad_time_list, metric_list, thread_ind=thread_ind)
    for hessian_ind, hessian_option in enumerate(hessian_option_list):
        uv = mesh_energy.NodalVars(m, 2)
        uv_path = os.path.join(base_path, user_model_name, hessian_option, 'UVs')
        numUVs = getNumUVs(uv_path)
        uv_data_0 = read_uv_data(uv_path, 0)
        uv.setVars(uv_data_0)
        uv_data_final = read_uv_data(uv_path, numUVs - 1)
        m_union = mesh_operations.concatenateMeshes([(uv_data_0.reshape(-1, 2), m.elements()), (uv_data_final.reshape(-1, 2), m.elements())])
        
        # create viewer
        v = viewer.Viewer(m_union, wireframe=True)
        em = MeshFEM.EmbeddedMesh(m, uv)
        v.update(mesh=em, preserveExisting=False)
        v.makeOpaque(color='#48B3FF')
        
        spf = 1 / fps
        total_time = aligned_timing_list[hessian_ind][-1]
        numFrames = int(math.ceil(total_time / spf))
        video_fn = user_model_name + '_' + hessian_option + '_symmdsUVopt' + '_thread' + str(thread_num_list[thread_ind]) + '.mp4'
        
        start_record_timer = time.time()
        v.recordStart(os.path.join(save_directory, video_fn), renderScale=8, outputScale=2, framerate=fps, lineWidthScale=0.25)
        for f in range(numFrames):
            frameTime = f * spf
            iterationForFrame = max(0, bisect.bisect_right(aligned_timing_list[hessian_ind], frameTime) - 1)
            uv.setVars(read_uv_data(uv_path, iterationForFrame))
            v.update()
        v.recordStop()
        elapsed_record_time = time.time() - start_record_timer
        print(f"[Viedo] {video_fn} recorded in {save_directory}. Recording Time: {elapsed_record_time:.4f} seconds.")
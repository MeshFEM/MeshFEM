'''
Modularized(Functionalized) scripts to generate figures and videos for each model

Author:  Xinzhuo (johnson) Hu
Created: 01/17/2025  10:30:52pm
'''

import os, sys
sys.path.append('../')
import MeshFEM, mesh, benchmark
import numpy as np
import time
import plot_video_utils


def gen_plots_videos(base_path, modelbase_path, plots_folder_name, videos_folder_name, thread_num_list=[0]):
    # Check if the result path exists
    if not os.path.exists(base_path):
        print(f"Error: The specified result path '{base_path}' does not exist.")
        sys.exit(1)

    # Check if the modelbase path exists
    if not os.path.exists(modelbase_path):
        print(f"Error: The specified model base path '{modelbase_path}' does not exist.")
        sys.exit(1)

    # List all files in the directory and filter for 3D model files
    model_files = [f for f in os.listdir(modelbase_path) if f.endswith(('.obj', '.msh', '.off'))]
    if not model_files:
        print("No 3D model files (*.obj, *.msh, *.off) found in the specified folder.")
        sys.exit(1)

    numModels = len(model_files)
    total_timer = time.time()
    for model_ind, model_file in enumerate(model_files):
        model_name = os.path.splitext(model_file)[0]  # Extract the model name without extension
        print(f"{model_ind}/{numModels} Model: {model_name} generation starts.")
        in_model_timer = time.time()
        # read benchmark data
        obj_grad_time_list = plot_video_utils.readConvergenceTimingData(os.path.join(base_path, model_name), thread_num_list)
        obj_list, grad_norm_list, hessian_projected_list, hessian_shifted_amount_list = plot_video_utils.readHessianData(os.path.join(base_path, model_name))
        uv_dist_list = plot_video_utils.readUVdist(os.path.join(base_path, model_name))

        plot_dir = os.path.join(base_path, plots_folder_name, model_name)
        if not os.path.exists(plot_dir):  os.makedirs(plot_dir)

        video_dir = os.path.join(base_path, videos_folder_name, model_name)
        if not os.path.exists(video_dir):  os.makedirs(video_dir)

        # generate no-timing related figures
        plot_video_utils.save_grad_iter_figure(grad_norm_list, hessian_projected_list, model_name, plot_dir)
        plot_video_utils.save_uv_dist_figure(uv_dist_list, hessian_projected_list, model_name, plot_dir)

        # generate timing related figures and videos
        numThreads = len(thread_num_list)
        for i in range(numThreads):
            plot_video_utils.save_obj_grad_time_figure(obj_grad_time_list, model_name, plot_dir, i, thread_num_list)
            plot_video_utils.gen_GradIter_videos(grad_norm_list, hessian_projected_list, obj_grad_time_list, model_name, video_dir, i, thread_num_list)
            plot_video_utils.gen_ObjIter_videos(obj_list, hessian_projected_list, obj_grad_time_list, model_name, video_dir, i, thread_num_list)
            plot_video_utils.gen_Param_videos(base_path, obj_list, obj_grad_time_list, model_name, modelbase_path, video_dir, i, thread_num_list)
        
        in_model_elapsed_time = time.time() - in_model_timer
        print(f"{model_ind}/{numModels} Model: {model_name} -- All plots and videos generation completed! Time: {in_model_elapsed_time:.4f} seconds.")
        print("-----------------------------------------------------------------------------------------------------------------------------------------")
    
    total_elapsed_time = time.time() - total_timer
    print(f"All Plots and Videos Generation for {base_path} Completed! Total Time: {total_elapsed_time:.4f} seconds.")
    print("**********************************************************************************************************************************")

if __name__ == "__main__":
    # Check if the script is provided with the required arguments
    if len(sys.argv) < 3:
        print("Usage: python generate_plots_videos.py <result_path> <modelbase_path> [<plots_folder_name>] [<videos_folder_name>]")
        sys.exit(1)

    # Parse command-line arguments
    result_path = sys.argv[1]
    modelbase_path = sys.argv[2]
    plots_folder_name = sys.argv[3] if len(sys.argv) > 3 else 'Figures'
    videos_folder_name = sys.argv[4] if len(sys.argv) > 4 else 'Videos'

    
    thread_num_list = [0]
    # thread_num_list = [1, 2, 4, 8, 16]
    # Run the function to perform experiments
    gen_plots_videos(result_path, modelbase_path, plots_folder_name, videos_folder_name, thread_num_list)

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


def gen_plots_videos(base_path, modelbase_path, plots_folder_name, videos_folder_name, hessian_option_list, video_flag=False, 
                     thread_num_list=[0]):
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
        print(f"{model_ind+1}/{numModels} Model: {model_name} generation starts.")
        in_model_timer = time.time()
        # read benchmark data
        obj_grad_time_list = plot_video_utils.readConvergenceTimingData(os.path.join(base_path, model_name), thread_num_list, hessian_option_list)
        obj_list, grad_norm_list, hessian_projected_list, hessian_shifted_amount_list, step_list, dd_list = plot_video_utils.readHessianData(os.path.join(base_path, model_name), hessian_option_list)
        uv_dist_list = plot_video_utils.readUVdist(os.path.join(base_path, model_name), hessian_option_list)

        plot_dir = os.path.join(base_path, plots_folder_name, model_name)
        if not os.path.exists(plot_dir):  os.makedirs(plot_dir)

        # generate no-timing related figures
        plot_video_utils.saveMetricIterFigure(grad_norm_list, hessian_projected_list, model_name, 'Grad', plot_dir, hessian_option_list=hessian_option_list)
        plot_video_utils.saveMetricIterFigure(obj_list, hessian_projected_list, model_name, 'Obj', plot_dir, hessian_option_list=hessian_option_list)
        plot_video_utils.saveMetricIterFigure(uv_dist_list, hessian_projected_list, model_name, 'UVdist', plot_dir, offset=1, hessian_option_list=hessian_option_list)
        plot_video_utils.saveMetricIterFigure(step_list, hessian_projected_list, model_name, 'Step', plot_dir, offset=1, hessian_option_list=hessian_option_list)
        plot_video_utils.saveMetricIterFigure(dd_list, hessian_projected_list, model_name, 'DD', plot_dir, offset=1, hessian_option_list=hessian_option_list)

        # generate timing related figures and videos
        numThreads = len(thread_num_list)
        for i in range(numThreads):
            plot_video_utils.save_obj_grad_time_figure(obj_grad_time_list, model_name, plot_dir, i, thread_num_list, hessian_option_list)
            if video_flag:
                # confirm video flag
                video_dir = os.path.join(base_path, videos_folder_name, model_name)
                if not os.path.exists(video_dir):  os.makedirs(video_dir)
                plot_video_utils.gen_GradIter_videos(grad_norm_list, hessian_projected_list, obj_grad_time_list, model_name, video_dir, i, thread_num_list, hessian_option_list)
                plot_video_utils.gen_ObjIter_videos(obj_list, hessian_projected_list, obj_grad_time_list, model_name, video_dir, i, thread_num_list, hessian_option_list)
                plot_video_utils.gen_Param_videos(base_path, obj_list, obj_grad_time_list, model_name, modelbase_path, video_dir, i, thread_num_list, hessian_option_list)
        
        in_model_elapsed_time = time.time() - in_model_timer
        print(f"{model_ind+1}/{numModels} Model: {model_name} -- All plots and videos generation completed! Time: {in_model_elapsed_time:.4f} seconds.")
        print("-----------------------------------------------------------------------------------------------------------------------------------------")
    
    total_elapsed_time = time.time() - total_timer
    print(f"All Plots and Videos Generation for {base_path} Completed! Total Time: {total_elapsed_time:.4f} seconds.")
    print("**********************************************************************************************************************************")

if __name__ == "__main__":
    # Check if the script is provided with the required arguments
    if len(sys.argv) < 3:
        print("Usage: python generate_plots_videos.py <result_path> <modelbase_path> <hessian_list_option> <thread_list_option> [<videos_flag>] [<plots_folder_name>] [<videos_folder_name>]")
        print("--------------------------------------------------------------------------------------------------------------------")
        print("Usage: Hessian Option List: [0] -- [Adaptive, Always, xbasedAlways]")
        print("Usage: Hessian Option List: [1] -- [Adaptive, Always, xbasedAlways, Never, TinyAD]")
        print("Usage: Hessian Option List: [2] -- [Adaptive, Always, xbasedAlways, TinyAD]")
        print("------------------------------------------------------------")
        print("Usage: Thread Option List: [0] -- [0](default thread)")
        print("Usage: Thread Option List: [1] -- [20]")     
        print("Usage: Thread Option List: [2] -- [2, 4, 8] ")   
        sys.exit(1)

    # Parse command-line arguments
    result_path = sys.argv[1]
    modelbase_path = sys.argv[2]
    hessian_list_option = int(sys.argv[3])
    thread_list_option = int(sys.argv[4])
    videos_flag = sys.argv[5] if len(sys.argv) > 5 else 'no'
    plots_folder_name = sys.argv[6] if len(sys.argv) > 6 else 'Figures'
    videos_folder_name = sys.argv[7] if len(sys.argv) > 7 else 'Videos'

    gen_video_flag = False
    if videos_flag == 'yes' or videos_flag == 'Yes' or videos_flag == 'YES':
        gen_video_flag = True
    
    hessian_option_list = ['Adaptive', 'Always', 'xbasedAlways', 'TinyAD']
    thread_num_list = [0]

    if hessian_list_option == 1:
        hessian_option_list = ['Adaptive', 'Always', 'xbasedAlways', 'Never', 'TinyAD']
    elif hessian_list_option == 2:
        hessian_option_list = ['Adaptive', 'Always', 'xbasedAlways', 'TinyAD']
    elif hessian_list_option == 0:
        hessian_option_list = ['Adaptive', 'Always', 'xbasedAlways']
    
    if thread_list_option == 1:  thread_num_list = [20]
    elif thread_list_option == 0: thread_num_list = [0]
    elif thread_list_option == 2: thread_num_list = [2, 4, 8]


    gen_plots_videos(result_path, modelbase_path, plots_folder_name, videos_folder_name, hessian_option_list, gen_video_flag, thread_num_list)

'''
Modularized(Functionalized) scripts to generate figures and videos for each model

Author:  Xinzhuo (johnson) Hu
Created: 01/17/2025  10:30:52pm
'''

import os, sys
sys.path.append('../')
import MeshFEM, mesh, benchmark
import argparse
import numpy as np
import time
import plot_video_utils

MODEL_BASE = '../../../Models/TableOneModels'  # Configure The Path If you Want to Generate Parametrization Videos!

# Used to determine how many iteration we show in the "energy and gradient norm vs time plot"
section_dict = {"armadilloDisc": 35, "armchairDisc": 35, "bear_cut": 30, "bimba100KDisc": 30, "bladeDisc": 50, "buddha_cut": 30,
                "bumpy_sphereDisc": 30, "bunnyBotschDisc": 40, "busteDisc": 35, "camille_hand100KDisc": 45, "chinese_dragon": 125,
                "dragonHead2": 35, "gargoyle_cut": 50, "hand": 35, "Superman_cut1": 35, "Superman_cut2": 30, "Superman_cut3": 30,
                "vase_lion": 50, "cow2Disc": 50, "davidDisc": 35, "deformed_armadilloDisc": 45, "denteDisc": 30, "eros": 30, "Lucy_3cuts": 80}

# For plotting ObjGradVSTime Plots
timesection_dict = {"armadilloDisc": 100, "armchairDisc": 80, "bear_cut": 80, "bimba100KDisc": 65, "bladeDisc": 70, "buddha_cut": 50,
                "bumpy_sphereDisc": 30, "bunnyBotschDisc": 200, "busteDisc": 110, "camille_hand100KDisc": 195, "chinese_dragon": 200,
                "dragonHead2": 80, "gargoyle_cut": 200, "hand": 50, "Superman_cut1": 50, "Superman_cut2": 40, "Superman_cut3": 40,
                "vase_lion": 125, "cow2Disc": 100, "davidDisc": 70, "deformed_armadilloDisc": 200, "denteDisc": 190, "eros": 65, "Lucy_3cuts": 200}

# For TinyAD Only
speedup_dict = {"bear_cut": 5, "buddha_cut": 5, "deformed_armadilloDisc": 5, "Superman_cut1": 5}

def read_non_comment_lines(txt_path):
    """
    Reads a text file line by line, ignoring lines that start with '#'.
    Returns a list of non-comment lines.
    """
    if not os.path.isfile(txt_path):
        print(f"File not found: {txt_path}")
        return []

    result_lines = []
    try:
        with open(txt_path, 'r') as file:
            for line in file:
                stripped = line.strip()
                if not stripped.startswith('#') and stripped != '':
                    result_lines.append(stripped)
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    return result_lines

def gen_plots_videos(base_path, modeltxt_path, plots_folder_name, videos_folder_name, hessian_option_list, video_flag=False, 
                     thread_num_list=None):
    # Check if the result path exists
    if not os.path.exists(base_path):
        print(f"Error: The specified result path '{base_path}' does not exist.")
        sys.exit(1)

    model_name_list = read_non_comment_lines(modeltxt_path)

    numModels = len(model_name_list)
    total_timer = time.time()
    for model_ind, model_name in enumerate(model_name_list):
        # model_name = os.path.splitext(model_file)[0]  # Extract the model name without extension
        print(f"{model_ind+1}/{numModels} Model: {model_name} generation starts.")
        in_model_timer = time.time()
        # read benchmark data
        obj_grad_time_list = plot_video_utils.readConvergenceTimingData(os.path.join(base_path, model_name), thread_num_list, hessian_option_list)
        obj_list, grad_norm_list, hessian_projected_list, hessian_shifted_amount_list, hessian_indef_list, step_list, dd_list = plot_video_utils.readHessianData(os.path.join(base_path, model_name), hessian_option_list)
        # uv_dist_list = plot_video_utils.readUVdist(os.path.join(base_path, model_name), hessian_option_list)
        model_dict = plot_video_utils.readDictData(os.path.join(base_path, model_name), thread_num_list, hessian_option_list)

        numThreads = len(thread_num_list)
        if video_flag:
            # we can only render videos when thread_num = 16
            for i in range(numThreads):
                if thread_num_list[i] != 16:  continue
                video_dir = os.path.join(base_path, videos_folder_name, model_name)
                if not os.path.exists(video_dir):  os.makedirs(video_dir)

                obj_sected_list = plot_video_utils.sectMetricList(obj_list, sect=section_dict[model_name])
                grad_norm_sected_list = plot_video_utils.sectMetricList(grad_norm_list, sect=section_dict[model_name])
                hessian_projected_sected_list = plot_video_utils.sectMetricList(hessian_projected_list, sect=section_dict[model_name])
                obj_grad_time_sected_list = plot_video_utils.sectObjGradTimeList(obj_grad_time_list, sect=section_dict[model_name])

                # full version
                # plot_video_utils.gen_MetricIter_videos(grad_norm_list, hessian_projected_list, obj_grad_time_list, 
                #                                        model_name, 'Grad', video_dir, i, thread_num_list, hessian_option_list, speedup=1)
                # plot_video_utils.gen_MetricIter_videos(obj_list, hessian_projected_list, obj_grad_time_list, 
                #                                        model_name, 'Obj', video_dir, i, thread_num_list, hessian_option_list, speedup=1)
                # plot_video_utils.gen_Param_videos(base_path, obj_list, obj_grad_time_list, 
                #                                     model_name, MODEL_BASE, video_dir, i, thread_num_list, hessian_option_list, speedup=1)
                
                # sected Version
                
                # plot_video_utils.gen_MetricIter_videos(obj_sected_list, hessian_projected_sected_list, obj_grad_time_sected_list, 
                #                                        model_name, 'Obj', video_dir, i, thread_num_list, hessian_option_list, speedup=1, sect=section_dict[model_name])

                plot_video_utils.gen_MetricIter_videos(grad_norm_sected_list, hessian_projected_sected_list, obj_grad_time_sected_list, 
                                                       model_name, 'Grad', video_dir, i, thread_num_list, hessian_option_list, speedup=1, sect=section_dict[model_name])
                
                # plot_video_utils.gen_Param_videos(base_path, obj_sected_list, obj_grad_time_sected_list, 
                #                                     model_name, MODEL_BASE, video_dir, i, thread_num_list, hessian_option_list, speedup=speedup_dict[model_name], sect=section_dict[model_name])
                # For TinyAD
                # plot_video_utils.gen_Param_videos(base_path, obj_sected_list, obj_grad_time_sected_list, 
                #                                     model_name, MODEL_BASE, video_dir, i, thread_num_list, hessian_option_list, speedup=1, sect=section_dict[model_name])
                
        
        else:
            plot_dir = os.path.join(base_path, plots_folder_name, model_name)
            if not os.path.exists(plot_dir):  os.makedirs(plot_dir)

            # generate no-timing related figures
            plot_video_utils.saveMetricIterFigure(grad_norm_list, hessian_projected_list, model_name, 'Grad', plot_dir, hessian_option_list=hessian_option_list)
            plot_video_utils.saveMetricIterFigure(grad_norm_list, hessian_projected_list, model_name, 'Grad', plot_dir, sect=timesection_dict[model_name], hessian_option_list=hessian_option_list, scName='ProjIndef', hessian_indef_list=hessian_indef_list) # for hessian projection scatter
            
            plot_video_utils.saveMetricIterFigure(obj_list, hessian_projected_list, model_name, 'Obj', plot_dir, sect=timesection_dict[model_name], hessian_option_list=hessian_option_list)
            plot_video_utils.saveMetricIterFigure(obj_list, hessian_projected_list, model_name, 'Obj', plot_dir, hessian_option_list=hessian_option_list)
            
            # Debug Only
            # plot_video_utils.saveMetricIterFigure(uv_dist_list, hessian_projected_list, model_name, 'UVdist', plot_dir, offset=1, hessian_option_list=hessian_option_list)
            # plot_video_utils.saveMetricIterFigure(step_list, hessian_projected_list, model_name, 'Step', plot_dir, offset=1, hessian_option_list=hessian_option_list)
            # plot_video_utils.saveMetricIterFigure(dd_list, hessian_projected_list, model_name, 'DD', plot_dir, offset=1, hessian_option_list=hessian_option_list)

            # generate bar plots
            metric_key_list = ['time', 'hessian_eval', 'linsolve']
            for metric_keyword in metric_key_list:
                plot_video_utils.saveMetricBarPlots(model_dict, model_name, metric_keyword, plot_dir, thread_num_list, hessian_option_list)
                plot_video_utils.saveMetricBarPlots(model_dict, model_name, metric_keyword, plot_dir, thread_num_list, hessian_option_list, divideIter=True)

            # # generate timing related figures and videos
            for i in range(numThreads):
                if thread_num_list[i] == 16: # only plot when thread = 16
                    plot_video_utils.save_obj_grad_time_figure(obj_grad_time_list, model_name, plot_dir, i, thread_num_list, hessian_option_list)
                    # plot_video_utils.save_obj_grad_time_figure_MiddleCut(obj_grad_time_list, model_name, plot_dir, i, thread_num_list, hessian_option_list)
                    plot_video_utils.save_obj_grad_time_figure(obj_grad_time_list, model_name, plot_dir, i, thread_num_list, hessian_option_list, sect=timesection_dict[model_name])
                    plot_video_utils.save_obj_grad_time_figure(obj_grad_time_list, model_name, plot_dir, i, thread_num_list, hessian_option_list, sect=100) # another 100
                    
            
        in_model_elapsed_time = time.time() - in_model_timer
        print(f"{model_ind+1}/{numModels} Model: {model_name} -- All plots and videos generation completed! Time: {in_model_elapsed_time:.4f} seconds.")
        print("-----------------------------------------------------------------------------------------------------------------------------------------")
    
    total_elapsed_time = time.time() - total_timer
    print(f"All Plots and Videos Generation for {base_path} Completed! Total Time: {total_elapsed_time:.4f} seconds.")
    print("**********************************************************************************************************************************")

def validate_videos_flag(value):
    """Validate that the videos_flag is 'yes', 'no', or 'both' (case-insensitive)."""
    valid_options = {"yes", "no"}
    if value.lower() not in valid_options:
        raise argparse.ArgumentTypeError(f"Invalid value for videos_flag: '{value}'. Must be one of {valid_options}.")
    return value.lower()  # Return the lowercase version for consistency

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Generate plots and videos for a given experiment setup."
    )
    parser.add_argument("result_path", type=str, help="Path to save the experiment results.")
    parser.add_argument("modeltxt_path", type=str, help="model name txt file path.")
    parser.add_argument(
        "-hessian_options",
        type=str,
        nargs="+",
        required=True,
        help="List of hessian options to test (e.g., -hessian_options Adaptive AutoDiff).",
    )
    parser.add_argument(
        "-threads",
        type=int,
        nargs="+",  # Accepts one or more integers
        required=True,
        help="List of thread numbers to use (e.g., -threads 1 2 4 8).",
    )
    parser.add_argument(
        "-videos_flag",
        type=validate_videos_flag,
        default="no",
        help="Option to generate videos: 'yes' or 'no' (default: 'no').",
    )
    parser.add_argument(
        "-plots_folder_name",
        type=str,
        default="Figures",
        help="Folder name to save plots (default: 'Figures').",
    )
    parser.add_argument(
        "-videos_folder_name",
        type=str,
        default="Videos",
        help="Folder name to save videos (default: 'Videos').",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Process parsed arguments
    hessian_option_list = args.hessian_options
    thread_num_list = args.threads  # Parsed as a list of integers
    gen_video_flag = args.videos_flag == "yes"  # Convert flag to boolean

    # Debugging information (optional)
    print(f"Using Hessian Options: {hessian_option_list}")
    print(f"Using Thread Numbers: {thread_num_list}")
    print(f"Generate Videos: {gen_video_flag}")
    print(f"Plots Folder: {args.plots_folder_name}")
    print(f"Videos Folder: {args.videos_folder_name}")

    # Call your main function
    gen_plots_videos(
        args.result_path,
        args.modeltxt_path,
        args.plots_folder_name,
        args.videos_folder_name,
        hessian_option_list,
        gen_video_flag,
        thread_num_list,
    )


if __name__ == "__main__":
    print("Usage: python generate_plots_videos.py <result_path> <modeltxt_path> <hessian_list_option> <thread_list_option> [<videos_flag>] [<plots_folder_name>] [<videos_folder_name>]")
    print("Supported Hessian Options: <Adaptive> <Always> <xbasedAlways> <AutoDiff> <AdaptiveAbs> <AutoDiffAbs> <TinyAD> <SLIM> <CompMajor>(Linux Only)")
    print("--------------------------------------------------------------------------------------------------------------------")
    
    main()

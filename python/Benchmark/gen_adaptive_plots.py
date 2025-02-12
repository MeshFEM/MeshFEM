'''
Modularized(Functionalized) scripts to generate plots for Adaptive parameter tuning experiments

Author:  Xinzhuo (johnson) Hu
Created: 02/11/2025  15:57:45pm
'''

import os, sys
sys.path.append('../')
import MeshFEM, mesh, benchmark
import argparse
import numpy as np
from pathlib import Path
from glob import glob
import time
import plot_video_utils


def get_stepTuple_list(consecutive_step_list, projection_step_list, include_default_adaptive=True):
    step_tuple_list = []
    if include_default_adaptive:  
        step_tuple_list.append((5, 10)) # default parameters in Adaptive
    
    for c_step in consecutive_step_list:
        for p_step in projection_step_list:
            step_tuple = (c_step, p_step)
            step_tuple_list.append(step_tuple)
            
    return step_tuple_list

def get_stepTupleStr_list(step_tuple_list):
    step_tuple_str_list = []
    for tuple in step_tuple_list:
        step_tuple_str = f"C{tuple[0]}P{tuple[1]}"
        step_tuple_str_list.append(step_tuple_str)

    return step_tuple_str_list


def validate_flag(value):
    """Validate that the flag is 'yes', 'no' (case-insensitive)."""
    valid_options = {"yes", "no"}
    if value.lower() not in valid_options:
        raise argparse.ArgumentTypeError(f"Invalid value for flag: '{value}'. Must be one of {valid_options}.")
    return value.lower()  # Return the lowercase version for consistency

def gen_adaptive_exp_plots(result_path, modelbase_path, thread_num_list, step_tuple_list, plots_folder_name):
    '''
    For each 3d model in modelbase_path, read exp data in the folder of result_path

    Under result_path, create another folder named 'plots_folder_name' to store plots of each model
    '''

    # Get all model files in the modelbase directory
    model_files = list(modelbase_path.glob("*.off")) + \
                  list(modelbase_path.glob("*.obj")) + \
                  list(modelbase_path.glob("*.msh"))

    if not model_files:
        raise FileNotFoundError(f"No 3D model files found in '{modelbase_path}'.")
        
    # Iterate over models
    numModels = len(model_files)
    total_timer = time.time()
    for model_ind, model_file in enumerate(model_files):
        model_name = model_file.stem
        model_result_path = result_path / model_name
        print(f"[Adaptive Experiment Plots] {model_ind+1}/{numModels} Model: {model_name} generateion starts.")
        # read hessian timing data
        adaptive_exp_data_dict = plot_video_utils.readHessianTimingData(str(model_result_path), thread_num_list, step_tuple_list)
        # read benchmark data
        model_bk_dict = plot_video_utils.readDictData(str(model_result_path), thread_num_list, step_tuple_list)

        # create Plots Folder
        save_plot_path = result_path / plots_folder_name / model_name
        save_plot_path.mkdir(parents=False, exist_ok=True)

        # Generate GradNorm vs Iter plots
        step_tuple_str_list = get_stepTupleStr_list(step_tuple_list)
        plot_video_utils.saveMetricIterFigure_AdapExpWrapper(adaptive_exp_data_dict, thread_num_list[-1], model_name,
                                                             'grad_norm_arr', str(save_plot_path), step_tuple_str_list, 
                                                             offset=0, sect=None, scName='ProjIndef', addHessianIndef=True)
        
        

        print(f"[Adaptive Experiment Plots] {model_ind+1}/{numModels} Model: {model_name} generateion completes!")
        print("-------------------------------------------------------------------------------------------------------")
    
    total_elapsed_time = time.time() - total_timer
    print(f"All Adaptive Experiment Plots Generation for {result_path} Completed! Total Time: {total_elapsed_time:.4f} seconds.")
    print("**********************************************************************************************************************************")



def main():
    """
    Parses command-line arguments and calls functions to generate plots.
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Generate Plots for Adaptive Parameter-Tuning Experiments.")
    parser.add_argument("result_path", type=str, help="Path to store results.")
    parser.add_argument("modelbase_path", type=str, help="Path containing 3D model files.")
    parser.add_argument(
        "-consecutive_steps",
        type=int,
        nargs="+",  # Accepts one or more integers
        required=True,
        help="List of thread numbers to use (e.g., -consecutive_steps 5).",
    )
    parser.add_argument(
        "-projection_steps",
        type=int,
        nargs="+",  # Accepts one or more integers
        required=True,
        help="List of thread numbers to use (e.g., -projection_steps 1 2 4 8 10).",
    )
    parser.add_argument(
        "-default_param_flag",
        type=validate_flag,
        default="yes",
        help="Option to generate videos: 'yes' or 'no' (default: 'no').",
    )
    parser.add_argument("-threads", type=int, nargs="+", default=[16], help="List of thread numbers in plots (default: 16).")
    parser.add_argument(
        "-plots_folder_name",
        type=str,
        default="Figures",
        help="Folder name to save plots (default: 'Figures').",
    )
    
    args = parser.parse_args()

    # Convert paths to Path objects
    result_path = Path(args.result_path)
    modelbase_path = Path(args.modelbase_path)
    consecutive_step_list = args.consecutive_steps
    projection_step_list = args.projection_steps
    thread_num_list = args.threads
    include_default_flag = args.default_param_flag == "yes"  # The default of Adaptive Option is C-5 P-10
    
    # Validate paths
    if not result_path.exists() or not result_path.is_dir():
        raise FileNotFoundError(f"Result path '{result_path}' does not exist or is not a directory.")

    if not modelbase_path.exists() or not modelbase_path.is_dir():
        raise FileNotFoundError(f"Model base path '{modelbase_path}' does not exist or is not a directory.")
    
    # Debugging information (optional)
    print(f"[Gen Adapative Plots]----------------------------------------------------------------------------------------")
    print(f"Consecutive Steps: {consecutive_step_list}")
    print(f"Projection Steps: {projection_step_list}")
    print(f"Using Thread Numbers: {thread_num_list}")
    print(f"Plot Default Parameter: {include_default_flag}")
    print(f"Plots Folder: {args.plots_folder_name}")

    step_tuple_list = get_stepTuple_list(consecutive_step_list, projection_step_list, include_default_flag)
    gen_adaptive_exp_plots(result_path, modelbase_path, thread_num_list, step_tuple_list, args.plots_folder_name)

if __name__ == "__main__":
    main()

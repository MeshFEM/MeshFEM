import os
import sys
import argparse
import subprocess
import numpy as np
import time
from datetime import datetime

def writelog(result_path, hessian_option_list, thread_num_list, save_uv_option, repeat_number):
    # Log file name
    log_file_name = 'experiment_log.txt'
    log_file_path = os.path.join(result_path, log_file_name)
    
    # Check if the log file exists, create it if not
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            pass  # Create an empty file
    
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Write to the log file
    with open(log_file_path, 'a') as log_file:
        # Write the timestamp
        log_file.write(f"Parametrization Benchmarking Experiment: {timestamp}\n")
        # Write the Hessian option list
        log_file.write(f"Hessian Option List: {hessian_option_list}\n")
        # Write the thread number list
        log_file.write(f"Thread Number List: {thread_num_list}\n")
        log_file.write(f"Save UV Option: {save_uv_option}\n")
        log_file.write(f"Repeat Number: {repeat_number}\n")

def run_all_models(result_path, modelbase_path, save_uv_option, repeat_num, thread_num_list, hessian_projection_labels):
    # Check if the result path exists
    if not os.path.exists(result_path):
        print(f"Error: The specified result path '{result_path}' does not exist.")
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

    # Iterate through all model files
    total_timer = time.time()
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]  # Extract the model name without extension
        model_path = os.path.join(modelbase_path, model_file)
        model_timer = time.time()

        option_time_list = []
        # Run the experiment for each Hessian Projection option
        for solver_option in hessian_projection_labels:
            print(f"\nStarting Parametrization for model '{model_name}' with Hessian Projection option '{solver_option}'...\n")
            hessian_option_timer = time.time()

            thread_time_list = []
            for thread_num in thread_num_list:
                thread_timer = time.time()
                if thread_num == 0:  print(f"[Opt] Parametrization using default thread number.")
                else:                print(f"[Opt] Parametrization using thread number - {thread_num}.")

                # runSymmDiriParam.py handles repeat
                cmd = [
                    sys.executable,  # Python executable path
                    "runSymmDiriParam.py",
                    result_path,
                    model_name,
                    model_path,
                    solver_option,
                    save_uv_option,
                    str(thread_num),
                    str(repeat_num) 
                ]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error during execution: {e}")
                    sys.exit(1)

                elapsed_thread_time = time.time() - thread_timer
                thread_time_list.append(elapsed_thread_time)
                print(f"Finished parametrization using thread number - {thread_num}. Time: {elapsed_thread_time : .4f} seconds.")

            option_time_list.append(thread_time_list)
            elapsed_hessian_option_time = time.time() - hessian_option_timer
            print(f"Completed Parametrization for model '{model_name}' with Hessian Projection option '{solver_option}'. Time: {elapsed_hessian_option_time : .4f} seconds.\n")
        
        elapsed_model_time = time.time() - model_timer
        option_time_list.append(elapsed_model_time)

        if save_uv_option.lower() == 'no':
            # save option_time_list only make sense when save_uv_option is no
            model_result_path = os.path.join(result_path, model_name)
            option_thread_time_fn = "option_thread_times.txt"
            with open(os.path.join(model_result_path, option_thread_time_fn), "w") as f:
                for i, label in enumerate(hessian_projection_labels):
                    f.write(f"{label}\t" + "\t".join(f"{x: .4f}" for x in option_time_list[i]) + "\n")
                f.write(f"\nTotal time:\t{option_time_list[-1]: .4f}\n")
            print(f"[File] Successfully Write '{option_thread_time_fn}' in {model_result_path}.")

        print(f"Completed All Parametrization Experiments for model '{model_name}'. Time: {elapsed_model_time : .4f} seconds.")
        print("-------------------------------------------------------------------------------------------------------------")
    
    elapsed_total_time = time.time() - total_timer
    print(f"\nAll experiments completed successfully! Total Time: {elapsed_total_time : .4f} seconds.")


def validate_save_uv_option(value):
    """Validate that the save_uv_option is 'yes', 'no', or 'both' (case-insensitive)."""
    valid_options = {"yes", "no", "both"}
    if value.lower() not in valid_options:
        raise argparse.ArgumentTypeError(f"Invalid value for save_uv_option: '{value}'. Must be one of {valid_options}.")
    return value.lower()  # Return the lowercase version for consistency

def main():
    # Define Hessian options mapping
    hessian_options_map = {
        0: ['Adaptive', 'Always', 'xbasedAlways'],
        1: ['Adaptive', 'Always', 'xbasedAlways', 'Never', 'TinyAD'],
        2: ['Adaptive', 'Always', 'xbasedAlways', 'TinyAD'],
        3: ['TinyAD'],
        4: ['Never'],
        5: ['Adaptive', 'Always', 'xbasedAlways', 'AutoDiff'],
        6: ['Adaptive', 'Always', 'xbasedAlways', 'AutoDiff', 'TinyAD'],
        7: ['AutoDiff'],
        8: ['SLIM'],
        9: ['Adaptive', 'Always', 'xbasedAlways', 'AutoDiff', 'TinyAD', 'SLIM']
    }

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Run all models for a given experiment setup with specified options."
    )
    parser.add_argument("result_path", type=str, help="Path to save the experiment results.")
    parser.add_argument("modelbase_path", type=str, help="Base path to the models.")
    parser.add_argument(
        "save_uv_option",
        type=validate_save_uv_option,  # Custom validation
        help="Option to save UV data: 'yes', 'no', or 'both' (case-insensitive).",
    )
    parser.add_argument(
        "hessian_list_option",
        type=int,
        choices=hessian_options_map.keys(),
        help="Choose a Hessian option list by index.",
    )
    parser.add_argument(
        "-threads",
        type=int,
        nargs="+",  # Accepts one or more integers
        required=True,
        help="List of thread numbers to use (e.g., -threads 1 2 4 8).",
    )
    parser.add_argument(
        "-repeat",
        type=int,
        default=1,
        help="Number of repetitions (default: 1). Must be a positive integer.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate repeat_num
    if args.repeat <= 0:
        print("[Error] <repeat> must be a positive integer >= 1.")
        sys.exit(1)

    # Get hessian_option_list and thread_num_list
    hessian_option_list = hessian_options_map.get(args.hessian_list_option, [])
    thread_num_list = args.threads  # Automatically parsed as a list of integers

    # Debugging information (optional)
    print(f"Using Save UV Option: {args.save_uv_option}")
    print(f"Using Hessian Options: {hessian_option_list}")
    print(f"Using Thread Numbers: {thread_num_list}")

    # Call your main functions
    run_all_models(
        args.result_path,
        args.modelbase_path,
        args.save_uv_option,
        args.repeat,
        thread_num_list,
        hessian_option_list,
    )
    writelog(
        args.result_path,
        hessian_option_list,
        thread_num_list,
        args.save_uv_option,
        args.repeat,
    )

if __name__ == "__main__":
    print("Usage: python run_all_models.py <result_path> <modelbase_path> <save_uv_option> <hessian_list_option> <thread_list_option> [<repeat_num>]")
    print("--------------------------------------------------------------------------------------------------------------------")
    print("Usage: Hessian Option List: [0] -- [Adaptive, Always, xbasedAlways]")
    print("Usage: Hessian Option List: [1] -- [Adaptive, Always, xbasedAlways, Never, TinyAD]")
    print("Usage: Hessian Option List: [2] -- [Adaptive, Always, xbasedAlways, TinyAD]")
    print("Usage: Hessian Option List: [3] -- [TinyAD]")
    print("Usage: Hessian Option List: [4] -- [Never]")
    print("Usage: Hessian Option List: [5] -- [Adaptive, Always, xbasedAlways, AutoDiff]")
    print("Usage: Hessian Option List: [6] -- [Adaptive, Always, xbasedAlways, AutoDiff, TinyAD]")
    print("Usage: Hessian Option List: [7] -- [AutoDiff]")
    print("Usage: Hessian Option List: [8] -- [SLIM]")
    print("Usage: Hessian Option List: [9] -- [Adaptive, Always, xbasedAlways, AutoDiff, TinyAD, SLIM]")
    main()
    

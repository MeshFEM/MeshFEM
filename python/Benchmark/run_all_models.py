import os
import sys
import argparse
import subprocess
import numpy as np
import MeshFEMParamSolverEnum as SolverOptionEnum
import time
from datetime import datetime

def writelog(result_path, model_files, hessian_option_list, thread_num_list, save_uv_option, repeat_number, hessian_shift):
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
        # Write the model files
        log_file.write(f"Model Filename List: {model_files}\n")
        log_file.write("\n")
        # Write the Hessian option list
        log_file.write(f"Hessian Option List: {hessian_option_list}\n")
        log_file.write(f"Hessian Shift of MeshFEM: {hessian_shift}\n")
        # Write the thread number list
        log_file.write(f"Thread Number List: {thread_num_list}\n")
        log_file.write(f"Save UV Option: {save_uv_option}\n")
        log_file.write(f"Repeat Number: {repeat_number}\n")

def run_all_models(result_path, modelbase_path, save_uv_option, repeat_num, thread_num_list, hessian_projection_labels, hessian_shift):
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
                    str(repeat_num),
                    str(hessian_shift) 
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

    # Write Log
    writelog(result_path, model_files, hessian_projection_labels, thread_num_list, save_uv_option, repeat_num, hessian_shift)


def validate_save_uv_option(value):
    """Validate that the save_uv_option is 'yes', 'no', or 'both' (case-insensitive)."""
    valid_options = {"yes", "no", "both"}
    if value.lower() not in valid_options:
        raise argparse.ArgumentTypeError(f"Invalid value for save_uv_option: '{value}'. Must be one of {valid_options}.")
    return value.lower()  # Return the lowercase version for consistency

# No return, just a check function
def validate_hessian_options(values):
    # avoid duplicates
    numOptions = len(values)
    if numOptions != len(set(values)):  raise argparse.ArgumentTypeError("Duplicated values in your hessian_options list.")

    # each option should be case-sensitive
    valid_hessian_option_list = ['MeshFEM', 'TinyAD', 'SLIM', 'CompMajor']
    for i in range(numOptions):
        if values[i] not in valid_hessian_option_list:
            raise argparse.ArgumentTypeError(f"Invalid value for hessian_options: '{values[i]}'. Must be one of {valid_hessian_option_list}.")
    
def validate_solver_varind_list(values, numMeshFEMVariants):
    # if values is a string
    if len(values) == 1:
        val = values[0].lower()
        if val == 'all':  return 'all'
        elif val == 'none':  return 'none'
    
    # Otherwise, interpret as list of integers
    try:
        int_list = [int(v) for v in values]
    except ValueError:
        raise argparse.ArgumentTypeError("All values must be integers or 'all'/'none'.")

    for x in int_list:
        if not (0 <= x < numMeshFEMVariants):
            raise argparse.ArgumentTypeError(f"Invalid value {x}: must be >= 0 and < {numMeshFEMVariants}.")
    
    # check for duplicates in int_list
    if len(int_list) != len(set(int_list)):  raise argparse.ArgumentTypeError("Duplicated values in your solver_varind_list.")
    
    return int_list

def composeSolverOptionList(hessian_options, solver_varind_list, numMeshFEMVariants):
    solver_options_list = []
    for option in hessian_options:
        if option == 'MeshFEM':
            if solver_varind_list == 'all': solver_options_list += [f'MeshFEM{n}' for n in range(numMeshFEMVariants)]
            elif solver_varind_list == 'none': continue
            else:
                for varind in solver_varind_list:  
                    solver_str = 'MeshFEM' + str(varind)
                    solver_options_list.append(solver_str)
        else:  solver_options_list.append(option)
    return solver_options_list


def main():
    numMeshFEMSettings = len(SolverOptionEnum.MeshFEMSettings())
    numMeshFEMVariants = 2 ** numMeshFEMSettings
    # --- Build help text dynamically ---
    variant_lines = []
    for i in range(numMeshFEMVariants):
        names = SolverOptionEnum.optionNamesFromIndex(i)
        line = f"  {i:2d}: {', '.join(names)}"
        variant_lines.append(line)
    variant_help_suffix = "\n".join(variant_lines)

    help_text = (
        "Specify 'all', 'none', or a space-separated list of integers in [0 ~ 2^n], each representing a solver variant of MeshFEM:\n"
        + variant_help_suffix
    )

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
        "-hessian_options",
        type=str,
        nargs="+",
        required=True,
        help="List of hessian options to test (e.g., -hessian_options MeshFEM TinyAD SLIM CompMajor).",
    )
    parser.add_argument(
        "-solver_varind_list",
        nargs='+',
        type=str,
        required=True,
        help=help_text,
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
    parser.add_argument(
        "-MeshFEM_hessian_shift",
        type=float,
        default=1e-12,
        help="Hessian shift amount set in MeshFEM's newton problem.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate repeat_num
    if args.repeat <= 0:
        print("[Error] <repeat> must be a positive integer >= 1.")
        sys.exit(1)

    # Prepare hessian_option_list 
    validated_var_list = validate_solver_varind_list(args.solver_varind_list, numMeshFEMVariants)
    validate_hessian_options(args.hessian_options)
    hessian_option_list = composeSolverOptionList(args.hessian_options, validated_var_list, numMeshFEMVariants)

    thread_num_list = args.threads  # Automatically parsed as a list of integers

    # Debugging information (optional)
    print(f"Using Save UV Option: {args.save_uv_option}")
    print(f"Using Hessian Options: {hessian_option_list}")
    print(f"Using Thread Numbers: {thread_num_list}")
    print(f"Using hessian shift in MeshFEM: {args.MeshFEM_hessian_shift}")

    # Call your main functions
    run_all_models(
        args.result_path,
        args.modelbase_path,
        args.save_uv_option,
        args.repeat,
        thread_num_list,
        hessian_option_list,
        args.MeshFEM_hessian_shift
    )

    
if __name__ == "__main__":
    print("Usage: python run_all_models.py <result_path> <modelbase_path> <save_uv_option> <hessian_options> <solver_varind_list> <threads> [<repeat>] [<MeshFEM_hessian_shift>]")
    # print("Supported Hessian Options: <Adaptive> <Always> <xbasedAlways> <AutoDiff> <AdaptiveAbs> <AutoDiffAbs> <TinyAD> <SLIM> <CompMajor> (Linux Only)")
    print("Supported Hessian Options: <MeshFEM+int(0~2^n)> <TinyAD> <SLIM> <CompMajor> (Linux Only)")
    print("--------------------------------------------------------------------------------------------------------------------")

    main()
    

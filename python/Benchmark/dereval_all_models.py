'''
Top-level Script to call runDerEvalTiming.py on all models
'''

import os, sys
import argparse
import subprocess
import time
from datetime import datetime

DICT_FILE_NAME = "total_timing_dict.pkl.gz"
# Run almost every combinations below 
Derivative_Type_List = ['AN', 'FAD', 'TAD']
Projection_Type_List = ['None', 'Fbased', 'Xbased']

def writelog(result_path, model_files, method_list, thread_num_list, repeat_number):
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
        log_file.write(f"Method List: {method_list}\n")
        # Write the thread number list
        log_file.write(f"Thread Number List: {thread_num_list}\n")
        log_file.write(f"Repeat Number: {repeat_number}\n")

def initialize_methodkey_in_dict(method_list, derivative_type_list, projection_type_list) -> list[str]:
    method_key_list = []
    if 'MeshFEM' in method_list:
        for derivative_type in derivative_type_list:
            for projection_type in projection_type_list:
                if derivative_type == 'TAD' and projection_type == 'Fbased': continue # invalid TAD and Fbased in MeshFEM
                method_key = f"MeshFEM-{derivative_type}-{projection_type}"
                method_key_list.append(method_key)

    if 'TinyAD' in method_list:
        derivative_type = 'None'
        for projection_type in projection_type_list:
            if projection_type == 'Fbased': continue  # invalid Fbased in TinyAD
            method_key = f"TinyAD-{derivative_type}-{projection_type}"
            method_key_list.append(method_key)
    
    return method_key_list

def initializeDictFile(result_path, model_name_list, method_list, thread_num_list):
    TotalTimingDict = {}
    method_key_list = initialize_methodkey_in_dict(method_list, Derivative_Type_List, Projection_Type_List)
    for model_name in model_name_list:
        model_dict = {}
        for method_key in method_key_list:
            method_dict = {}
            for thread_num in thread_num_list:
                method_dict[thread_num] = 0.0  # initialize as 0 
            model_dict[method_key] = method_dict
        TotalTimingDict[model_name] = model_dict
    
    import helper_funcs
    dict_file_path = os.path.join(result_path, DICT_FILE_NAME)
    helper_funcs.save_dict(TotalTimingDict, dict_file_path)
    print(f"[FILE] {DICT_FILE_NAME} Initialized Successfully in {result_path}.")

def run_der_evaluations(result_path, modelbase_path, method_list, thread_num_list, repeat_num):
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
    
    # model_names_list
    model_names = [os.path.splitext(model_file)[0] for model_file in model_files]

    # Initialization Dict File in result_path
    initializeDictFile(result_path, model_names, method_list, thread_num_list)

    # run Eval On All models
    start_timer = time.perf_counter()
    for model_file in model_files:
        model_timer = time.perf_counter()
        model_path = os.path.join(modelbase_path, model_file)
        # for each method, combined with derivative_type and projection_type
        for method in method_list:
            method_timer = time.perf_counter()

            if method == 'MeshFEM':  temp_derivative_type_list = Derivative_Type_List
            else:  temp_derivative_type_list = ['None']
            for derivative_type in temp_derivative_type_list:
                for projection_type in Projection_Type_List:
                    # skip unreasonable combinations
                    if method == 'TinyAD' and projection_type == 'Fbased': continue
                    if method == 'MeshFEM' and derivative_type == 'TAD' and projection_type == 'Fbased': continue
                    for thread_num in thread_num_list:
                        cmd = [
                            sys.executable,
                            "runDerEvalTiming.py",
                            result_path,
                            model_path,
                            method,
                            derivative_type,
                            projection_type,
                            str(thread_num),
                            str(repeat_num)
                        ]
                        try:
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Error during execution: {e}")
                            sys.exit(1)
            method_elapsed_total_time = time.perf_counter() - method_timer
            print("**************")
            print(f"Completed Method - {method} evaluation for model {model_file}. Total Time: {method_elapsed_total_time : .8f} seconds.")
        model_elapsed_total_time = time.perf_counter() - model_timer
        print("-------------------------------------------------------------------------------------------------")
        print(f"Completed All Method Evaluation for model {model_file}. Total Time: {model_elapsed_total_time : .8f} seconds.")

    elapsed_total_time = time.perf_counter() - start_timer
    print(f"\nAll experiments completed successfully! Total Time: {elapsed_total_time : .4f} seconds.")
    writelog(result_path, model_files, method_list, thread_num_list, repeat_num)


# No return, just a check function
def validate_methods(values):
    # avoid duplicates
    numOptions = len(values)
    if numOptions != len(set(values)):  
        raise argparse.ArgumentTypeError("Duplicated values in your method list.")

    # each option should be case-sensitive
    valid_method_list = ['MeshFEM', 'TinyAD']
    for i in range(numOptions):
        if values[i] not in valid_method_list:
            raise argparse.ArgumentTypeError(f"Invalid value for hessian_options: '{values[i]}'. Must be one of {valid_method_list}.")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Eval all models on a list of methods evaluating their derivative eval times."
    )
    parser.add_argument("result_path", type=str, help="Path to save the experiment results.")
    parser.add_argument("modelbase_path", type=str, help="Base path to the models.")
    parser.add_argument(
        "-method_list",
        type=str,
        nargs="+",
        required=True,
        help="List of hessian options to test (e.g., -hessian_options MeshFEM TinyAD).",
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
        default=10,
        help="Number of repetitions (default: 1). Must be a positive integer.",
    )

    # Parse the arguments
    args = parser.parse_args()
    # Validate repeat_num
    if args.repeat <= 0:
        print("[Error] <repeat> must be a positive integer >= 1.")
        sys.exit(1)
    # Validate method list
    method_list = args.method_list
    thread_num_list = args.threads  # Automatically parsed as a list of integers
    repeat_num = args.repeat

    # Debugging information (optional)
    print(f"Using Methods: {method_list}")
    print(f"Using Thread Numbers: {thread_num_list}")
    print(f"Using Repeat: {repeat_num}")

    # Run Full Evaluations
    run_der_evaluations(args.result_path, args.modelbase_path, method_list, thread_num_list, repeat_num)

if __name__ == "__main__":
    print("Usage: python dereval_all_models.py <result_path> <modelbase_path> <method_list> <threads> [<repeat>]")
    print("Supported Methods: <MeshFEM> <TinyAD>. And we will run all admissible combinations of derivative types and projection types.")
    print("--------------------------------------------------------------------------------------------------------------------")

    main()
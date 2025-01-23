import os
import sys
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

if __name__ == "__main__":
    # Check if the script is provided with the required arguments
    if len(sys.argv) < 6:
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
        print("------------------------------------------------------------")
        print("Usage: Thread Option List: [0] -- [0](default thread)")
        print("Usage: Thread Option List: [1] -- [16]")
        print("Usage: Thread Option List: [2] -- [1] ")
        print("Usage: Thread Option List: [3] -- [4, 8, 16] ")
        print("Usage: Thread Option List: [4] -- [2] ")
        print("Usage: Thread Option List: [5] -- [2, 4, 8, 16] ")
        print("Usage: Thread Option List: [6] -- [1, 2, 4, 8, 16] ")
        sys.exit(1)

    # Parse command-line arguments
    result_path = sys.argv[1]
    modelbase_path = sys.argv[2]
    save_uv_option = sys.argv[3]
    hessian_list_option = int(sys.argv[4])
    thread_list_option = int(sys.argv[5])
    repeat_num = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    if repeat_num <= 0:
        print("[Error] <repeat_num> must be an positive integer >= 1.")
        sys.exit(1)
    
    thread_num_list = [0]
    hessian_option_list = ['Adaptive', 'Always', 'xbasedAlways']

    if hessian_list_option == 1:
        hessian_option_list = ['Adaptive', 'Always', 'xbasedAlways', 'Never', 'TinyAD']
    elif hessian_list_option == 2:
        hessian_option_list = ['Adaptive', 'Always', 'xbasedAlways', 'TinyAD']
    elif hessian_list_option == 3:
        hessian_option_list = ['TinyAD']
    elif hessian_list_option == 4:
        hessian_option_list = ['Never']
    elif hessian_list_option == 5:
        hessian_option_list = ['Adaptive', 'Always', 'xbasedAlways', 'AutoDiff']
    elif hessian_list_option == 6:
        hessian_option_list = ['Adaptive', 'Always', 'xbasedAlways', 'AutoDiff', 'TinyAD']
    elif hessian_list_option == 7:
        hessian_option_list = ['AutoDiff']
    
    if thread_list_option == 1:  thread_num_list = [16]
    elif thread_list_option == 2:  thread_num_list = [1]
    elif thread_list_option == 5:  thread_num_list = [2, 4, 8, 16]
    elif thread_list_option == 3:  thread_num_list = [4, 8, 16]
    elif thread_list_option == 4:  thread_num_list = [2]
    elif thread_list_option == 6:  thread_num_list = [1, 2, 4, 8, 16]
    
    run_all_models(result_path, modelbase_path, save_uv_option, repeat_num, thread_num_list, hessian_option_list)
    writelog(result_path, hessian_option_list, thread_num_list, save_uv_option, repeat_num)
    

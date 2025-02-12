import os, sys
sys.path.append('../')
import MeshFEM, mesh, benchmark
import parallelism
import numpy as np
import pickle
import helper_funcs
import argparse
from pathlib import Path
from glob import glob
import time
from datetime import datetime

def writelog(result_path_obj, modelbase_path, consecutive_step_list, projection_step_list, thread_nums, repeat_num):
    # Log file name
    log_file_name = 'experiment_log.txt'
    log_file_path = result_path_obj / log_file_name
    
    # Check if the log file exists, create it if not
    if not log_file_path.exists():
        with log_file_path.open("w") as f:
            pass  # Create an empty file
    
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Write to the log file
    with log_file_path.open("a") as log_file:
        # Write the timestamp
        log_file.write(f"Adaptive Parameter Tuning Experiment: {timestamp}\n")
        # Write the Hessian option list
        log_file.write(f"modelbase_path: {str(modelbase_path.absolute())}\n")
        # Write the thread number list
        log_file.write(f"Consecutive Step List: {consecutive_step_list}\n")
        log_file.write(f"Projection Step List: {projection_step_list}\n")
        log_file.write(f"Thread Number List: {thread_nums}\n")
        log_file.write(f"Repeat Number: {repeat_num}\n")
        log_file.write("---------------------------------------------------------------------------------------")


def saveOptData(path_obj, obj_arr, time_arr, grad_norm_arr,
                benchmark_dict, hessian_projected_arr, hessian_shift_arr,
                hessian_indef_arr, step_size_arr, dd_arr):
    
    if (obj_arr.shape[0] != time_arr.shape[0]):
        raise RuntimeWarning("[File] Array size mismatch of objective array and time array")
    if (obj_arr.shape[0] != grad_norm_arr.shape[0]):
        raise RuntimeWarning("[File] Array size mismatch of objective array and gradient norm array")
    
    arr_fn = 'obj_time_gradnorm.npz'
    dict_fn = 'benchmark_dict.pkl'
    hessian_data_fn = 'hessian_data.npz'
    save_arr_path = path_obj / arr_fn
    save_dict_path = path_obj / dict_fn
    save_hessian_data_path = path_obj / hessian_data_fn

    np.savez_compressed(save_arr_path, obj_arr = obj_arr, time_arr = time_arr, grad_norm_arr=grad_norm_arr)
    np.savez_compressed(save_hessian_data_path, hp_arr = hessian_projected_arr, hs_arr = hessian_shift_arr, hi_arr = hessian_indef_arr,
                        step_size_arr = step_size_arr, dd_arr = dd_arr)
    # save benchmark dictionary
    with save_dict_path.open("wb") as f:
        pickle.dump(benchmark_dict, f)
    
    print(f"[File] Successfully Write {arr_fn}, {dict_fn}, {hessian_data_fn}, in {str(path_obj)}!")

def optuv_models(result_path, modelbase_path, consecutive_step_list, projection_step_list, thread_nums, repeat_num):
    """
    Processes 3D models, creates necessary directories, and runs `runSymmds_AdaptiveParameter`.

    Args:
        result_path (Path): Path where results are stored.
        modelbase_path (Path): Path where 3D models are located.
        thread_nums (list[int]): List of thread numbers to iterate over.
        repeat_num (int): Number of times to repeat the function execution.
    """
    # Get all model files in the modelbase directory
    model_files = list(modelbase_path.glob("*.off")) + \
                  list(modelbase_path.glob("*.obj")) + \
                  list(modelbase_path.glob("*.msh"))

    if not model_files:
        raise FileNotFoundError(f"No 3D model files found in '{modelbase_path}'.")

    # Iterate over models
    total_timer = time.time()
    for model_file in model_files:
        # Read 3D model data
        mesh_data = helper_funcs.read_mesh(str(model_file))

        model_name = model_file.stem  # Extract filename without extension
        model_result_path = result_path / model_name
        print(f'[Adaptive Experiment] Symmetric Dirichlet Parametrization of Model: {model_name}')        
        # Create model folder if it doesn't exist
        model_result_path.mkdir(parents=False, exist_ok=True)

        # Iterate over numCISBE and numPSBD
        for numCISBE in consecutive_step_list:
            for numPSBD in projection_step_list:
                print(f'[Adaptive Experiment] Number Consecutive Indefinite Steps Before Before Enable: {numCISBE}.')
                print(f'[Adaptive Experiment] Number Projection Steps Before Disable: {numPSBD}.')
                print('------------------------------------------------------------------------------------------------------------')

                cxpy_folder = model_result_path / f"C{numCISBE}P{numPSBD}"
                cxpy_folder.mkdir(parents=False, exist_ok=True)

                # Iterate over thread numbers
                for thread_num in thread_nums:
                    thread_folder = cxpy_folder / f"thread_{thread_num}"
                    thread_folder.mkdir(parents=False, exist_ok=True)
                    parallelism.set_max_num_tbb_threads(int(thread_num))
                    print(f'[Adaptive Experiment] Thread Number {thread_num}.')

                    # Statistics List
                    total_time_list = []  # Used for searching the quickest experiment
                    final_obj_list = []  
                    newton_steps_list = []
                    symbolic_factorize_time_list = []
                    numeric_factroize_time_list = []
                    linsys_solve_time_list = []
                    hessian_eval_time_list = []

                    # Run function multiple times
                    for i in range(repeat_num):
                        print(f"Running parametrization experiment {i + 1}/{repeat_num}...")
                        repeat_folder = thread_folder / f"repeat_{i+1}"
                        repeat_folder.mkdir(parents=False, exist_ok=True)

                        obj_arr, time_arr, grad_norm_arr, benchmark_dict, \
                        hessian_projected_arr, hessian_shifted_arr, hessian_indef_arr, \
                        step_size_arr, dd_arr = helper_funcs.runSymmds_AdaptiveParameter(mesh_data, numCISBE, numPSBD, max_iter=200, hessian_shift=1e-8)
                        
                        symbolic_factorize_time = benchmark.totalTime('Catamari Symbolic Factorize$', d=benchmark_dict)
                        numeric_factorize_time = benchmark.totalTime('Catamari Numeric Factorize$', d=benchmark_dict)
                        linsys_solve_time = benchmark.totalTime('CholeskyFactorizerBase.solve$', d=benchmark_dict) + symbolic_factorize_time + numeric_factorize_time
                        hessian_eval_time = benchmark.totalTime('NewtonMultiobjectiveProblem.hessian$', d=benchmark_dict) + benchmark.totalTime('NewtonMultiobjectiveProblem.gradient$', d=benchmark_dict)
                        symbolic_factorize_time_list.append(symbolic_factorize_time)
                        numeric_factroize_time_list.append(numeric_factorize_time)

                        newton_steps = obj_arr.shape[0]
                        total_time = time_arr[-1]
                        final_obj = obj_arr[-1]

                        total_time_list.append(total_time)
                        final_obj_list.append(final_obj)
                        newton_steps_list.append(newton_steps)
                        linsys_solve_time_list.append(linsys_solve_time)
                        hessian_eval_time_list.append(hessian_eval_time)
                        
                        print(f"[Opt] Symmetric Dirichlet Parametrization of {model_name} Ended in {newton_steps} Newton Steps. Total Elapsed Time: {total_time: .4f} seconds.")
                        saveOptData(repeat_folder, obj_arr, time_arr, grad_norm_arr, benchmark_dict,
                                    hessian_projected_arr, hessian_shifted_arr, hessian_indef_arr,
                                    step_size_arr, dd_arr)
                        
                        print(f"Ended parametrization experiment {i + 1}/{repeat_num}.")
                    
                    # write to summary.txt file
                    summary_txt_fn = 'summary.txt'
                    summary_save_filepath = thread_folder / summary_txt_fn
                    summary_data = np.column_stack((newton_steps_list, total_time_list, final_obj_list, symbolic_factorize_time_list,
                                    numeric_factroize_time_list, linsys_solve_time_list, hessian_eval_time_list))
                    np.savetxt(summary_save_filepath, summary_data, fmt='%.8f', delimiter="\t", 
                               header="iter\t\ttime\t\tenergy\t\tsymbol\t\tnumeric\t\tlinsolve\thessian_eval", comments='')

                    # Find the fastest one and add it to 'summary.txt'
                    total_time_min = min(total_time_list)
                    index_min = total_time_list.index(total_time_min)
                    with summary_save_filepath.open("a") as f:
                        f.write(f"Fastest: {index_min + 1}\n")
                    
                    print(f'[Adaptive Experiment] Experiment Completed! Model: {model_name}. Consecutive Step: {numCISBE}. Projection Step: {numPSBD}. Thread Number {thread_num}.')    

                print('------------------------------------------------------------------------------------------------------------')
    
    elapsed_total_time = time.time() - total_timer
    print(f"\n [Adaptive Experiment] All experiments completed successfully! Total Time: {elapsed_total_time : .4f} seconds.")

def main():
    """
    Parses command-line arguments and calls `optuv_models`.
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run Adaptive Parameters on 3D Models.")
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
    parser.add_argument("-threads", type=int, nargs="+", default=[16], help="List of thread numbers to use (default: 16).")
    parser.add_argument("-repeat", type=int, default=1, help="Number of repetitions (default: 1).")

    args = parser.parse_args()

    # Convert paths to Path objects
    result_path = Path(args.result_path)
    modelbase_path = Path(args.modelbase_path)
    consecutive_step_list = args.consecutive_steps
    projection_step_list = args.projection_steps

    # Validate paths
    if not result_path.exists() or not result_path.is_dir():
        raise FileNotFoundError(f"Result path '{result_path}' does not exist or is not a directory.")

    if not modelbase_path.exists() or not modelbase_path.is_dir():
        raise FileNotFoundError(f"Model base path '{modelbase_path}' does not exist or is not a directory.")
    
    os.environ['OMP_NUM_THREADS'] = '1'
    # Call the processing function
    optuv_models(result_path, modelbase_path, consecutive_step_list, projection_step_list, args.threads, args.repeat)
    writelog(result_path, modelbase_path, consecutive_step_list, projection_step_list, args.threads, args.repeat)

if __name__ == "__main__":
    main()

'''
Run Derivative Evaluation on Surface Parameterization for one model
Aim: a Comprehensive Evaluation comparing TinyAD's derivative evaluation overhead 

Author: Xinzhuo (Johnson) Hu
Created: 11/06/2025  10:23:38 PM

'''

import os, sys
sys.path.append('../')
import warnings
from pathlib import Path

DICT_FILE_NAME = "total_timing_dict.pkl.gz"

def runEvalTiming(base_path, model_path, method, derivative_type, projection_type, thread_num, repeat):
    from helper_funcs import read_mesh, derivativeEvalTiming, load_dict, save_dict
    model_name = Path(model_path).stem
    # Print the parameters for confirmation
    print("-------------------------------------------------------------------------------------------------------------------")
    print(f"Running Symmetric Dirichelt Parametrization with the following parameters:")
    print(f"  Base Path: {base_path}")
    print(f"  Model Name: {model_name}")
    print(f"  Method: {method}")
    print(f"  Derivative Type: {derivative_type}")
    print(f"  Projection Type: {projection_type}")
    print(f"  Thread Number: {thread_num}")
    print(f"  Repeat: {repeat}")

    m = read_mesh(model_path) # read mesh from model_path
    perEvalTiming = derivativeEvalTiming(m, method, derivative_type, projection_type, repeat)

    # Read Dict file and Update and Save
    dict_file_path = os.path.join(base_path, DICT_FILE_NAME)
    total_timing_dict = load_dict(dict_file_path)
    method_key_str = f"{method}-{derivative_type}-{projection_type}"
    total_timing_dict[model_name][method_key_str][thread_num] = perEvalTiming
    save_dict(total_timing_dict, dict_file_path)

    print(f"[Timing] Evaluation of {method} with {derivative_type} and {projection_type} on thread-{thread_num} done on {model_name}! perEval Timing: {perEvalTiming : .8f} seconds")

def main():
    # Ensure at least 5 arguments (excluding script name) are provided
    if len(sys.argv) < 7:
        print("Usage: python runDerEvalTiming.py <base_path> <model_path> <method> <derivative_type> <projection_type> <thread_num> [<repeat>]")
        sys.exit(1)

    # Parse input arguments
    base_path = sys.argv[1]
    model_path = sys.argv[2]
    method = sys.argv[3]
    derivative_type = sys.argv[4]
    projection_type = sys.argv[5]
    thread_num = int(sys.argv[6])
    repeat = int(sys.argv[7]) if len(sys.argv) > 7 else 10

    if method not in ['TinyAD', 'MeshFEM']:
        print(f"[Error] Usage of <method>: {method} not Supported! ")
        sys.exit(1)
    
    if projection_type not in ['None', 'Fbased', 'Xbased']:
        print(f"[Error] Usage of <projection_type>: {projection_type} not Supported! ")
        sys.exit(1)

    if (method == 'MeshFEM') and (derivative_type not in ['AN', 'FAD', 'TAD']):
        print(f"[Error] Usage of <derivative_type>: {derivative_type} not Supported when using method {method}! ")
        sys.exit(1)

    # Check if base_path exists
    if not os.path.exists(base_path):
        warnings.warn(f"Warning: The base_path '{base_path}' does not exist. Please check the path.")
        sys.exit(1)  # Exit if the path does not exist

    # Check if dict_file exists in base_path
    dict_file_path = os.path.join(base_path, DICT_FILE_NAME)
    if not os.path.exists(dict_file_path):
        warnings.warn(f"Warning: The Dict File '{dict_file_path}' does not exist. Please check the file.")
        sys.exit(1) 

    # Set Correct Threading Environment
    if method in ['TinyAD']:
        os.environ['OMP_NUM_THREADS'] = str(thread_num)
        print(f"[Debug] Check Threading: {os.environ['OMP_NUM_THREADS']}.")
    else:
        import MeshFEM
        import parallelism
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        parallelism.set_max_num_tbb_threads(int(thread_num))

    runEvalTiming(base_path, model_path, method, derivative_type, projection_type, thread_num, repeat)

if __name__ == "__main__":
    main()

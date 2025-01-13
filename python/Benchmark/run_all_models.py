import os
import sys
import subprocess

def run_all_models(result_path, modelbase_path, repeat_num):
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
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]  # Extract the model name without extension
        model_path = os.path.join(modelbase_path, model_file)

        # Run the experiment for each Hessian Projection option
        for solver_option in ['Adaptive', 'Always', 'Never']:
            print(f"\nStarting Parametrization for model '{model_name}' with Hessian Projection option '{solver_option}'...\n")

            # runSymmDiriParam.py handles repeat
            cmd = [
                sys.executable,  # Python executable path
                "runSymmDiriParam.py",
                result_path,
                model_name,
                model_path,
                solver_option,
                str(repeat_num) 
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during execution: {e}")
                sys.exit(1)

            print(f"Completed Parametrization for model '{model_name}' with Hessian Projection option '{solver_option}'.\n")

    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    # Check if the script is provided with the required arguments
    if len(sys.argv) != 4:
        print("Usage: python run_all_models.py <result_path> <modelbase_path> <repeat_num>")
        sys.exit(1)

    # Parse command-line arguments
    result_path = sys.argv[1]
    modelbase_path = sys.argv[2]
    repeat_num = int(sys.argv[3])
    if repeat_num <= 0:
        print("[Error] <repeat_num> must be an positive integer >= 1.")
        sys.exit(1)

    # Run the function to perform experiments
    run_all_models(result_path, modelbase_path, repeat_num)

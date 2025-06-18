import os
import sys
import re

def rename_files(folder_path):
    if "CompMajor" not in folder_path:
        raise RuntimeError(f"The folder path does not contain 'CompMajor': {folder_path}")

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    pattern = re.compile(r'^uv_ravel_iter_(\d+)\.npz$')
    rename_map = {}

    # First pass: collect renaming targets
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            i = int(match.group(1))
            new_filename = f"uv_ravel_iter_{i+1}.npz"
            temp_filename = f"__tmp__{filename}"
            old_path = os.path.join(folder_path, filename)
            temp_path = os.path.join(folder_path, temp_filename)
            final_path = os.path.join(folder_path, new_filename)
            rename_map[temp_path] = final_path
            os.rename(old_path, temp_path)  # Step 1: rename to temp
            print(f"Temporarily renamed: {filename} -> {temp_filename}")

    # Second pass: rename from temp to final
    for temp_path, final_path in rename_map.items():
        if os.path.exists(final_path):
            print(f"Warning: Skipping rename to {os.path.basename(final_path)} — file already exists.")
        else:
            os.rename(temp_path, final_path)
            print(f"Renamed to final: {os.path.basename(temp_path)} -> {os.path.basename(final_path)}")

if __name__ == "__main__":
    print("[WARNING] For renaming *uv_ravel_iter_i.npz* files under CompMajor's UV Folder Only !!!")
    if len(sys.argv) != 2:
        print("Usage: python rename_uv_files.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    rename_files(folder_path)

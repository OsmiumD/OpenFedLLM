import os
import shutil


def delete_single_file_folders(root_folder):
    """
    Iterate through all subfolders in a folder and delete any folder containing only one file.

    Args:
        root_folder (str): Path to the root folder.
    """
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        # Check if the folder contains exactly one file and no subdirectories
        if len(filenames) == 1 and not dirnames:
            print(f"Deleting folder: {dirpath} (contains only one file: {filenames[0]})")
            shutil.rmtree(dirpath)


# Usage
root_folder = "./output"  # Replace with your folder path
delete_single_file_folders(root_folder)
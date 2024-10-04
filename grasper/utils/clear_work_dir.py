import os
import shutil

def delete_files_in_directory(directory_path):
    """
    删除指定目录下的所有文件。
    
    :param directory_path: 要删除文件的目录路径。
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) 
                print(f"Deleted directory and its contents: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def clear_directories(images_dir):
    base_dir = os.path.dirname(images_dir)
    if base_dir != "./data":
        base_dir = os.path.dirname(base_dir)
    masks_dir = os.path.join(base_dir, "masks")
    poses_dir = os.path.join(base_dir, "poses")
    temp_dir = os.path.join(base_dir, "temp")

    delete_files_in_directory(images_dir)
    delete_files_in_directory(masks_dir)
    delete_files_in_directory(poses_dir)
    delete_files_in_directory(temp_dir)

if __name__ == "__main__":
    clear_directories(images_dir="./data/images")
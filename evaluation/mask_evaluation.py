
import os
import open3d as o3d
import argparse
import multiprocessing as mp
import time
import torch
import numpy as np
import cv2
import re
import socket
import pickle
import psutil
import sys
sys.path.append("/home/descfly/6d_pose/mast3r-grasp")
from env.pybullet_sim import OjaPick
from get_pcd.pcds import run_get_pcd, get_obj_pcd, visiualize_pcd, get_world_mask_pts, rotation_matrix_to_euler_angles
from get_mask_track.track import mask_tracker, get_mask_bbox
from get_mask_track.utils.get_mask_reshape import get_mask_reshape
from grasper.start_camera.start_camera import UR5Camera
from grasper.utils.clear_work_dir import clear_directories
from grasper.utils.grasp_utils import save_joint_angles_to_file, get_joint_kinematics_tradition, save_pts_to_file
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def worker_function(func, args, queue):
    result = func(*args)  
    queue.put(result) 

def setup_multiprocessing():
    mp.set_start_method('spawn', force=True)

def is_pixel_in_mask(pixel_x, pixel_y, mask_image):
    if 0 <= pixel_x < mask_image.shape[1] and 0 <= pixel_y < mask_image.shape[0]:
        if mask_image[pixel_y, pixel_x] > 0:
            return True
    return False

def run_once_evaluation(image_path, mask_path, track_len, prompt, pixel_x, pixel_y):
    mask_tracker(dataset_path=image_path, save_dir=mask_path, text=prompt, image_len=track_len)
    mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith('.png')], key=natural_sort_key)
    
    mask_files_path = os.path.join(mask_path, mask_files[0])
    mask_image = cv2.imread(mask_files_path, cv2.IMREAD_GRAYSCALE) 

    if is_pixel_in_mask(pixel_x, pixel_y, mask_image):
        print("The pixel is in the mask")
        return True
    else:
        print(pixel_x, pixel_y, mask_image.shape)
        cv2.circle(mask_image, (pixel_x, pixel_y), radius=5, color=(0, 0, 255), thickness=-1)
        save_path = "./results/mask_not_in_mask.png"
        cv2.imwrite(save_path, mask_image) 
        print("The pixel is not in the mask")
        return False

def log_memory_usage():
    # 打印 GPU 内存使用情况
    if torch.cuda.is_available():
        print(f"GPU 空闲内存: {torch.cuda.memory_reserved(0) / 1e6} MB")
    
    # 打印 CPU 内存使用情况
    process = psutil.Process(os.getpid())
    print(f"CPU 内存使用: {process.memory_info().rss / 1e6} MB")

def main():
    setup_multiprocessing()
    parser = argparse.ArgumentParser(description="mast3r grasp")

    parser.add_argument("--image_path", type=str, required=True, help="image path for your rgb image")
    parser.add_argument("--mask_path", type=str, required=True, help="a path to save the mask")
    parser.add_argument("--track_len", type=int, help="the length of the track")
    parser.add_argument("--prompt", type=str, default=None, help="the text prompt to compare with the masks")
    parser.add_argument("--test_time", type=int, default=10, help="the number of times to run the evaluation")

    args = parser.parse_args()
    
    #get image in here
    env_sim = OjaPick(image_len=args.track_len, prompt=args.prompt)
    correct = 0
    for i in range(args.test_time):
        pixel_x, pixel_y = env_sim.env_sim()

        prompt = env_sim.get_prompt()
        print(f"Running evaluation {i+1} of {args.test_time}")
        log_memory_usage()
        if run_once_evaluation(args.image_path, args.mask_path, args.track_len, prompt, pixel_x, pixel_y):
            correct += 1
        env_sim.remove_object()
        # clear_directories(images_dir=args.image_path)
    test_accuracy = correct / args.test_time
    print(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()


           
    


import os
import open3d as o3d
import argparse
import multiprocessing as mp
import time
import torch
import numpy as np
import cv2
import socket
import pickle
from env.pybullet_sim import OjaPick
from get_pcd.pcds import run_get_pcd, get_obj_pcd, visiualize_pcd, get_world_mask_pts, rotation_matrix_to_euler_angles
from get_mask_track.track import mask_tracker, get_mask_bbox
from get_mask_track.utils.get_mask_reshape import get_mask_reshape
from grasper.start_camera.start_camera import UR5Camera
from grasper.utils.clear_work_dir import clear_directories
from grasper.utils.grasp_utils import save_joint_angles_to_file, get_joint_kinematics_tradition, save_pts_to_file
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def worker_function(func, args, queue):
    result = func(*args)  
    queue.put(result) 

def setup_multiprocessing():
    mp.set_start_method('spawn', force=True)

def main():
    setup_multiprocessing()
    parser = argparse.ArgumentParser(description="mast3r grasp")

    parser.add_argument("--image_path", type=str, required=True, help="image path for your rgb image")
    parser.add_argument("--mask_path", type=str, required=True, help="a path to save the mask")
    parser.add_argument("--track_len", type=int, help="the length of the track")
    parser.add_argument("--prompt", type=str, default=None, help="the text prompt to compare with the masks")
    parser.add_argument("--robot_host", type=str, default="192.168.1.161", help="the host of the robot")
    parser.add_argument("--grasp_type", type=str, default="anygrasp", help="the type of grasp to use")

    args = parser.parse_args()
    
    #get image in here
    env_sim = OjaPick(image_len=args.track_len, prompt=args.prompt)
    env_sim.env_sim()
    exit()
    prompt = env_sim.get_prompt()
    
    args1 = (args.image_path, args.mask_path, prompt, args.track_len)
    args2 = (args.image_path, args.track_len)

    queue1 = mp.Queue()
    queue2 = mp.Queue()

    process_get_mask = mp.Process(target=worker_function, args=(mask_tracker, args1, queue1))
    process_get_cloud = mp.Process(target=worker_function, args=(run_get_pcd, args2, queue2))
   
    process_get_mask.start()
    process_get_cloud.start()
     
    result_get_mask = queue1.get()  
    result_get_cloud = queue2.get()  
    pts_list = result_get_cloud[0]
    colors_list = result_get_cloud[1]
    shape_list = result_get_cloud[2]
    camera_to_world = result_get_cloud[3]
    mask_list = result_get_mask
    mask_list = get_mask_reshape(mask_list)
  
    mask_pts, mask_colors = get_obj_pcd(pts_list, colors_list, shape_list, mask_list)

    world_to_camera = np.linalg.inv(camera_to_world)
    camera_mask_pts  = get_world_mask_pts(mask_pts, world_to_camera)

    if args.grasp_type == "anygrasp":
        save_pts_to_file(camera_mask_pts, mask_colors, camera_to_world)
        return 

    else :
        from grasper.grasp import grasper
        translation_gg_to_base, rotation_gg_to_base, gripper_width, gg = grasper(camera_mask_pts, mask_colors, camera_to_world)

        if translation_gg_to_base is None:
            print("No valid grasp found")
            process_get_mask.join()
            process_get_cloud.join()
            visiualize_pcd(camera_mask_pts, mask_colors, gg=False)
            clear_directories(args.image_path)
            return
        else :
            visiualize_pcd(camera_mask_pts, mask_colors, gg)
            process_get_mask.join()
            process_get_cloud.join()
            env_sim.grasp_sim(translation=translation_gg_to_base, rotation=rotation_gg_to_base, width=gripper_width)
            clear_directories(args.image_path)
            return
    

if __name__ == "__main__":
    main()


           
    


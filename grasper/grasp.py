import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
 
import torch

import transforms3d
 
import pyrealsense2 as rs
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys 

# from anygrasp.grasp_detection.grasp_detector import grasp_obj
from graspnetAPI import GraspGroup
 
sys.path.append('./grasper/anygrasp/graspnet-baseline')
sys.path.append('./grasper/anygrasp/graspnet-baseline/utils')
from models.graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
 
 
def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load("./grasper/anygrasp/graspnet-baseline/checkpoint-kn.tar")
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    # set model to eval mode
    net.eval()
    return net

def get_world_to_camera(mask_pts, mask_colors, camera_to_world):
    world_to_camera = np.linalg.inv(camera_to_world)
    num_points = mask_pts.shape[0]
    homogeneous_points = np.hstack((mask_pts, np.ones((num_points, 1))))
    transformed_points = (world_to_camera @ homogeneous_points.T).T
    mask_pts = transformed_points[:, :3]
    mask_colors = mask_colors
    return mask_pts, mask_colors    

def get_and_process_data(mask_pts, mask_colors, camera_to_world):

    # cloud_masked, color_masked = get_world_to_camera(mask_pts, mask_colors, camera_to_world)
    cloud_masked = mask_pts
    color_masked = mask_colors
    # sample points
    num_point = 20000
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
 
    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled
 
    return end_points, cloud


def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg
 
def collision_detection(gg, cloud):
    voxel_size = 0.01
    collision_thresh = 0.01
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg


def vis_grasps(gg, cloud, num_top_grasps=50):
    gg.nms()
    gg.sort_by_score(reverse=True)  # Sort the grasps in descending order of scores
    gg = gg[:num_top_grasps]  # Keep only the top num_top_grasps grasps
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def transform_gg_to_wrist3(gg, camera_translation_wrist3, camera_rotation_wrist3):
    """
    Transform the best grasp pose from the camera coordinate system to the wrist3 coordinate system.
    """
    gg.nms()
    gg.sort_by_score(reverse=True)  # Sort the grasps in descending order of scores
    if len(gg) == 0:
        return None, None, None
    best_gg = gg[0]  # Keep only the top grasp

    # Extract translation and rotation matrix from the best grasp
    translation = best_gg.translation
    rotation_matrix = best_gg.rotation_matrix
    
    # Convert rotation matrix to a quaternion
    gg_rotation_quat = R.from_matrix(rotation_matrix).as_quat()
    
    # Convert gg rotation to a Rotation object
    gg_rotation = R.from_quat(gg_rotation_quat)
    
    # Convert camera rotation to a Rotation object if it is a quaternion
    if len(camera_rotation_wrist3) == 4:
        camera_rotation_wrist3 = R.from_quat(camera_rotation_wrist3)
    
    # Compute the inverse of the camera rotation (from camera to wrist3)
    camera_to_wrist3_rotation = camera_rotation_wrist3.inv()
    
    # Compute the transformation of gg's rotation from camera to wrist3
    wrist3_rotation = camera_to_wrist3_rotation * gg_rotation
    
    # Compute the transformed translation from camera to wrist3
    gg_translation = np.array(translation)
    camera_translation_wrist3 = np.array(camera_translation_wrist3)
    
    wrist3_translation = camera_to_wrist3_rotation.apply(gg_translation) + camera_translation_wrist3
    
    return wrist3_translation, wrist3_rotation.as_quat(),best_gg.width

def transform_gg_to_world(gg, camera_translation_world, camera_rotation_world):
    """
    Transform the best grasp pose from the camera coordinate system to the wrist3 coordinate system.
    """
    gg.nms()
    gg.sort_by_score(reverse=True)  # Sort the grasps in descending order of scores
    if len(gg) == 0:
        return None, None, None
    best_gg = gg[0]  # Keep only the top grasp

    # Extract translation and rotation matrix from the best grasp
    translation = best_gg.translation
    gg_rotation = best_gg.rotation_matrix
     
    # Compute the transformation of gg's rotation from camera to wrist3
    rotation = camera_rotation_world @ gg_rotation
    
    # Compute the transformed translation from camera to wrist3
    gg_translation = np.array(translation)
    
    translation = camera_rotation_world @ gg_translation + camera_translation_world
    
    return translation, rotation, best_gg.width

def grasper(mask_pts, mask_colors, camera_to_world, use_anygrasp=False, transform_to_world=True):
    if use_anygrasp:
        print('use_anygrasp')   
    else: 
        net = get_net()
        end_points, cloud = get_and_process_data(mask_pts, mask_colors, camera_to_world)
        gg = get_grasps(net, end_points)
        collision_thresh = 0.01
        if collision_thresh > 0:
            gg = collision_detection(gg, np.array(cloud.points))

    if transform_to_world:
        camera_translation_world = camera_to_world[:3, -1]
        camera_rotation_world = camera_to_world[:3, :3]
        translation_gg, rotation_gg, gripper_width = transform_gg_to_world(gg, camera_translation_world, camera_rotation_world)
    else:
        camera_translation_tool0 = [-0.02172549093517455, -0.09257370479885837, 0.052393691719884845]
        camera_rotation_tool0 = [ 0.005016083776298267,-0.027338364341427224, 0.02561907613034145, 0.9992853024421564]
        translation_gg, rotation_gg, gripper_width = transform_gg_to_wrist3(gg, camera_translation_tool0, camera_rotation_tool0)

    return translation_gg, rotation_gg, gripper_width, gg


#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
from utils.general_utils import PILtoTorch, PILtoTorch_depth
from utils.graphics_utils import fov2focal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

WARNED = False


def global_pose_to_world(global_pose, s, R, t):
    global_pose_to_world = []
    for pose in global_pose:
        t_ = s * R @ pose[:3, 3] + t   
        R_ = R @ pose[:3, :3]  
        new_pose = np.eye(4)
        new_pose[:3, :3] = R_
        new_pose[:3, 3] = t_
        global_pose_to_world.append(new_pose)
    return global_pose_to_world

def draw_camera(ax, R, t, scale=0.01, axis_scale=0.2, color='b'):
    """
    绘制一个表示相机位姿的简单相机模型和视锥体。
    
    :param ax: matplotlib 3D 坐标轴对象
    :param R: 3x3 旋转矩阵，表示相机的方向
    :param t: 3x1 平移向量，表示相机的位置
    :param scale: 相机模型的缩放因子
    :param color: 相机模型的颜色
    """
    # 相机原点
    origin = t.reshape(3)
    
    # 相机的四个角点（假设相机视锥体为金字塔形状）
    corners = np.array([
        [0, 0, 0],        # 相机的中心点
        [-1, -1, 2],      # 视锥体的四个角点
        [1, -1, 2],
        [1, 1, 2],
        [-1, 1, 2]
    ]) * scale
    
    # 旋转相机的角点以适应当前的相机方向
    corners = R @ corners.T
    corners = corners.T + origin

    # 绘制相机的视锥体
    for i in range(1, 5):
        ax.plot([origin[0], corners[i, 0]], [origin[1], corners[i, 1]], [origin[2], corners[i, 2]], color=color)
    
    # 绘制视锥体的边缘
    ax.plot([corners[1, 0], corners[2, 0]], [corners[1, 1], corners[2, 1]], [corners[1, 2], corners[2, 2]], color=color)
    ax.plot([corners[2, 0], corners[3, 0]], [corners[2, 1], corners[3, 1]], [corners[2, 2], corners[3, 2]], color=color)
    ax.plot([corners[3, 0], corners[4, 0]], [corners[3, 1], corners[4, 1]], [corners[3, 2], corners[4, 2]], color=color)
    ax.plot([corners[4, 0], corners[1, 0]], [corners[4, 1], corners[1, 1]], [corners[4, 2], corners[1, 2]], color=color)

    # 绘制相机位置
    ax.scatter(*origin, color=color, s=50)
        # 绘制相机坐标轴
    # ax.quiver(*origin, *R[:, 0], length=axis_scale, color='r')  # X轴
    # ax.quiver(*origin, *R[:, 1], length=axis_scale, color='g')  # Y轴
    # ax.quiver(*origin, *R[:, 2], length=axis_scale, color='b')  # Z轴


def visualize_camera_poses(camera_poses, gt_pose):
    """
    可视化多组相机位姿，并使用不同的颜色区分不同的相机。
    
    :param camera_poses: 每个相机的位姿列表，包含多个相机的位姿，每个相机可能有多个位姿
    :param camera_colors: 不同相机的颜色列表
    """
    camera_colors = ['b', 'r']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.5, 0])
    ax.set_ylim([-0.5, 0])
    ax.set_zlim([0.5, 1])
    for i, pose in enumerate(camera_poses):
        if i <= 2:
            R = pose[:3, :3]
            t = pose[:3, 3]
            draw_camera(ax, R, t, scale=0.03, axis_scale=0.8, color=camera_colors[0])
            ax.text(t[0], t[1], t[2], f'Cam mast3r_to_world Pose {i+1}', color=camera_colors[0])
    
    for i, pose in enumerate(gt_pose):
        if i<= 2 :
            R = pose[:3, :3]
            t = pose[:3, 3]
            draw_camera(ax, R, t, scale=0.03, axis_scale=0.8, color=camera_colors[1])
            ax.text(t[0], t[1], t[2], f'Cam gt Pose {i+1}', color=camera_colors[1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    save_path = "./results/camera_poses.png"
    plt.savefig(save_path)
   
def compute_relative_pose(pose_0, pose_i):
    pose_0_inv = np.linalg.inv(pose_0)
    
    relative_pose = np.dot(pose_0_inv, pose_i)
    
    return relative_pose


import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('./grasper')
sys.path.append('./grasper/utils')
from Kinematics import UR5Kinematics
from robotics_fk_ik import create_ur5_robot, compute_inverse_kinematics, compute_forward_kinematics


def save_joint_angles_to_file(filename, joint_angles):
    """
    将关节角度保存到文件中，每次写入直接写到下一行。

    :param filename: 保存关节角度的文件名
    :param joint_angles: 包含关节角度的列表，例如 [angle1, angle2, angle3, ...]
    """
    with open(filename, 'w') as file:  # 使用 'w' 模式打开文件进行追加
        # 将关节角度列表转换为字符串，每个角度用空格分隔
        angles_str = ' '.join(map(str, joint_angles))
        # 将字符串写入文件，并在末尾添加换行符
        file.write(angles_str + '\n')

def save_pts_to_file(mask_pts, mask_colors, camera_to_world, dir_name="./data/temp/"):
    mask_path = dir_name + "mask_pts.npy"
    mask_color_path = dir_name + "mask_colors.npy"
    np.save(mask_path, mask_pts)
    np.save(mask_color_path, mask_colors)
    np.save(dir_name + "camera_to_world.npy", camera_to_world)

def get_joint_kinematics_tradition(xyz, rotation_matrix):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz
    
    ur5 = create_ur5_robot()
    joint_angles = compute_inverse_kinematics(ur5, transformation_matrix)
    return joint_angles

def get_xyzrpy_kinematics(joint_angles):
    joint_angles = np.array(joint_angles)
    ur5 = create_ur5_robot()
    xyzrpy = compute_forward_kinematics(ur5, joint_angles)
    return xyzrpy


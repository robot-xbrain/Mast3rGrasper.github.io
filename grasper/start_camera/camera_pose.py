import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("./grasper/start_camera")
from ur5_circle import UR5Controller

class UR5Kinematics:
    def __init__(self):
        # 定义每个关节的DH参数
        self.dh_params = [
            {'theta': 0, 'd': 0.089159, 'a': 0, 'alpha': np.pi/2},
            {'theta': 0, 'd': 0, 'a': -0.42500, 'alpha': 0},
            {'theta': 0, 'd': 0, 'a': -0.39225, 'alpha': 0},
            {'theta': 0, 'd': 0.10915, 'a': 0, 'alpha': np.pi/2},
            {'theta': 0, 'd': 0.09465, 'a': 0, 'alpha': -np.pi/2},
            {'theta': 0, 'd': 0.0823, 'a': 0, 'alpha': 0},
        ]
        self.camera_translation_tool0 = np.array([-0.02172549093517455, -0.09257370479885837, 0.052393691719884845])
        self.camera_rotation_tool0 = R.from_quat([ 0.005016083776298267,-0.027338364341427224, 0.02561907613034145, 0.9992853024421564])

    def dh_transform_matrix(self, theta, d, a, alpha):
        """使用DH参数生成变换矩阵"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        """使用给定的关节角计算末端相对于基座的变换矩阵"""
        T = np.eye(4)  
        for i in range(6):
            theta = joint_angles[i] + self.dh_params[i]['theta']  
            d = self.dh_params[i]['d']
            a = self.dh_params[i]['a']
            alpha = self.dh_params[i]['alpha']
            T_i = self.dh_transform_matrix(theta, d, a, alpha)
            T = np.dot(T, T_i)
        
        return T

    def get_camera_pose(self, joint_angles):
        """根据给定的关节角度计算相机相对于基座的位姿"""
        T_tool0_base = self.forward_kinematics(joint_angles)
        T_camera_tool0 = np.eye(4)
        T_camera_tool0[:3, :3] = self.camera_rotation_tool0.as_matrix()
        T_camera_tool0[:3, 3] = self.camera_translation_tool0
        T_camera_base = np.dot(T_tool0_base, T_camera_tool0)

        return T_camera_base

    def get_camera_pose_sim(self, joint_angles):
        """根据给定的关节角度计算相机相对于基座的位姿"""
        T_tool0_base = self.forward_kinematics(joint_angles)

        T_camera_tool0 = np.eye(4)
        T_camera_tool0[:3, :3] = self.camera_rotation_tool0.as_matrix()
        T_camera_tool0[:3, 3] = self.camera_translation_tool0
        T_camera_base = np.dot(T_tool0_base, T_camera_tool0)
        T = np.eye(4)
        T[0, 0] = -1
        T[1, 1] = -1
        T_camera_base = np.dot(T, T_camera_base)
        return T_camera_base




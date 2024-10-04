import numpy as np
import roboticstoolbox as rtb
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R

class UR5Kinematics:
    def __init__(self):
        self.dh_params = [
            {'theta': 0, 'd': 0.089159, 'a': 0, 'alpha': np.pi/2},
            {'theta': 0, 'd': 0, 'a': -0.42500, 'alpha': 0},
            {'theta': 0, 'd': 0, 'a': -0.39225, 'alpha': 0},
            {'theta': 0, 'd': 0.10915, 'a': 0, 'alpha': np.pi/2},
            {'theta': 0, 'd': 0.09465, 'a': 0, 'alpha': -np.pi/2},
            {'theta': 0, 'd': 0.0823, 'a': 0, 'alpha': 0},
        ]
        self.ur5 = rtb.models.UR5()

    def dh_transform(self, a, alpha, d, theta):
        """返回DH变换矩阵"""
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        """计算正向运动学，返回末端执行器的4x4齐次变换矩阵"""
        a = [0, -0.425, -0.392, 0, 0, 0]
        alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
        d = [0.089, 0, 0, 0.109, 0.094, 0.082]

        T = np.eye(4)
        for i in range(6):
            T_i = self.dh_transform(a[i], alpha[i], d[i], joint_angles[i])
            T = np.dot(T, T_i)
            
        return T

    def inverse_kinematics(self, target_pose):
        """计算逆向运动学，根据目标末端位姿计算关节角度"""
        if target_pose.shape != (6,):
            target_position = target_pose[:3, -1]
            target_rotation = R.from_matrix(target_pose[:3, :3]).as_euler('xyz')
            target_pose = np.concatenate([target_position, target_rotation])
        def equations(joint_angles):
            T = self.forward_kinematics(joint_angles)
            target_position = target_pose[:3]
            target_rotation = target_pose[3:]

            current_position = T[:3, 3]
            current_rotation = R.from_matrix(T[:3, :3]).as_euler('xyz')

            return np.concatenate([
                current_position - target_position,
                current_rotation - target_rotation
            ])
        
        initial_guess = [0, 0, 0, 0, 0, 0]
        solution = fsolve(equations, initial_guess)
        return solution


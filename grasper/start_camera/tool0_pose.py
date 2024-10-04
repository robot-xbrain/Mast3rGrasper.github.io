import numpy as np
from scipy.spatial.transform import Rotation as R
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



import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def rotation_matrix_z(theta):
    """计算绕z轴的旋转矩阵"""
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    return Rz

def construct_homogeneous_matrix(rotation_matrix, translation_vector):
    """将旋转矩阵和位移向量拼接成一个 4x4 的齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation_vector
    return T

def create_ur5_robot():
    """创建 UR5 机械臂的 DH 参数模型"""
    # UR5 机械臂的 DH 参数
    links = [
        RevoluteDH(a=0, alpha=np.pi/2, d=0.089159, offset=0),
        RevoluteDH(a=-0.425, alpha=0, d=0, offset=0),
        RevoluteDH(a=-0.392, alpha=0, d=0, offset=0),
        RevoluteDH(a=0, alpha=np.pi/2, d=0.109, offset=0),
        RevoluteDH(a=0, alpha=-np.pi/2, d=0.094, offset=0),
        RevoluteDH(a=0, alpha=0, d=0.082  , offset=0)
    ]
    
    ur5 = DHRobot(links, name='UR5')
    return ur5

def compute_forward_kinematics(robot, joint_angles):
    """计算正向运动学"""
    T = robot.fkine(joint_angles)
    return T

def compute_inverse_kinematics(robot, target_pose):
    """计算逆向运动学"""
    solution = robot.ikine_LM(target_pose)
    return solution

def plot_robot(robot, joint_angles):
    """可视化机器人模型"""
    robot.plot(joint_angles)
    plt.show()
    plt.pause(10) 

def main():
    ur5 = create_ur5_robot()

    target_position = [-0.68794041 , 0.06714196 , 0.06664764]
    Rz = [[ 0.31406241, -0.36731293 ,-0.87546902],
         [-0.18368456, -0.9282117  , 0.32354743],
         [-0.93146381,  0.05919606 ,-0.35898614]]
    target_pose = construct_homogeneous_matrix(Rz, target_position)
    solution = compute_inverse_kinematics(ur5, target_pose)
    angles = solution.q  
    print("Inverse Kinematics Joint Angles (q):")
    print(angles)

    T = compute_forward_kinematics(ur5, angles)
    print("Forward Kinematics Transformation Matrix:")
    print(T)

    plot_robot(ur5, angles)



import numpy as np
import time
import rtde_control
import rtde_receive


class UR5Controller:
    def __init__(self, robot_host):
        self.robot_host = robot_host
        self.rtde_c = rtde_control.RTDEControlInterface(robot_host)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_host)
        self.stop_move = False

    def get_current_pose(self):
        return self.rtde_r.getActualTCPPose()

    def get_current_joints(self):
        """获取当前的关节角度。"""
        return self.rtde_r.getActualQ()
    
    def move_to_joints(self, target_joints, speed=0.3, acceleration=0.3):
        """
        使机器人移动到给定的关节角度。
        
        :param target_joints: 目标关节角度的列表或数组。
        :param speed: 运动速度，默认值为0.1。
        :param acceleration: 运动加速度，默认值为0.1。
        """
        self.rtde_c.moveJ(target_joints, speed, acceleration)
        while True:
            current_joints = self.get_current_joints()
            if all(abs(c - t) < 0.01 for c, t in zip(current_joints, target_joints)):
                break
            time.sleep(0.01)
    

    def move_to_pose(self, target_pose, speed=0.1, acceleration=0.1):
        self.rtde_c.moveL(target_pose, speed, acceleration)
        while True:
            current_pose = self.get_current_pose()
            if all(abs(c - t) < 0.001 for c, t in zip(current_pose[:3], target_pose[:3])):
                break
            time.sleep(0.01)
    
    def move_along_y(self, distance=0.4, speed=0.2, acceleration=0.1):
        current_pose = self.get_current_pose()
        
        target_pose = current_pose.copy()
        target_pose[1] -= distance 
        
        self.rtde_c.moveL(target_pose, speed, acceleration)

        while True:
            current_pose = self.get_current_pose()
            if all(abs(c - t) < 0.001 for c, t in zip(current_pose[:3], target_pose[:3])):
                break
            time.sleep(0.01)

    def circular_motion(self):
        # txt_file = "./grasper/start_camera/straight_line_joints.txt"
        txt_file = "./grasper/start_camera/circle_joints.txt"
        
        joint_data = []
        with open(txt_file, 'r') as file:
            for line in file:
                joint_angles = [float(value) for value in line.split()]
                joint_data.append(joint_angles)
    
        for joints in joint_data:
            self.rtde_c.moveJ(joints, speed=0.4, acceleration=0.4)  
            while True:
                current_joints = self.rtde_r.getActualQ()
                if all(abs(c - t) < 0.01 for c, t in zip(current_joints, joints)):
                    break
                time.sleep(0.01)
                if self.stop_move:
                    break
            if self.stop_move:
                break
    
    def disconnect(self):
        self.rtde_c.disconnect()




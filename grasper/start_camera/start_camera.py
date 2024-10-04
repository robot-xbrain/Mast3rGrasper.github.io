import pyrealsense2 as rs
import cv2
import os
import threading
import rtde_receive
import time
import numpy as np
import sys
sys.path.append("./grasper/start_camera")
from camera_pose import UR5Kinematics
import shutil
from ur5_circle import UR5Controller
from scipy.spatial.transform import Rotation as R


class UR5Camera:
    def __init__(self, robot_host, image_path, image_len, pcd_experiments=False, mask_experiments=False):
        self.robot_host = robot_host
        self.controller = UR5Controller(robot_host)
        self.pcd_experiments = pcd_experiments
        self.mask_experiments = mask_experiments

        self.start_joint = [-29.94*np.pi/180,-86.55*np.pi/180,68.86*np.pi/180,-67*np.pi/180,-76.5*np.pi/180,161*np.pi/180]
        self.target_joint = [25.67*np.pi/180,-81.52*np.pi/180,64.83*np.pi/180,-64.37*np.pi/180,-110.02*np.pi/180,198*np.pi/180]

        # 初始化Realsense相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
      
         # 尝试启动流，最多重试3次
        for attempt in range(3):
            try:
                if attempt > 0:
                    self.pipeline.stop()
                    time.sleep(2) 

                self.pipeline.start(self.config)
                break  
            except RuntimeError as e:
                if attempt == 2:
                    raise e  # 如果尝试3次后仍然失败，抛出异常
                time.sleep(2)  # 等待2秒钟再重试
        self.controller.rtde_c.moveJ(self.start_joint, speed=0.5, acceleration=0.5)
        
        self.device = self.pipeline.get_active_profile().get_device()
        self.color_sensor = self.device.query_sensors()[1] 

        self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)  
        self.color_sensor.set_option(rs.option.exposure, 200) 
        self.color_sensor.set_option(rs.option.gain, 16) 
        self.color_sensor.set_option(rs.option.white_balance, 4600)  
 
        # 创建保存图像的目录
        self.image_dir = image_path
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        
        # 创建保存关节角度的目录
        base_dir = os.path.dirname(image_path)
        base_dir = os.path.dirname(base_dir)
        pose_path = os.path.join(base_dir, "poses")
        self.joints_dir = pose_path

        self.depth_dir = os.path.join(base_dir, "depths")

        if not os.path.exists(self.joints_dir):
            os.makedirs(self.joints_dir)

        self.frame_count = 0
        self.image_len = image_len
        self.keep_running = True
        self.capture_signal = threading.Event()  
        self.move_signal = threading.Event()  

        # 创建并启动图像采集线程
        self.move_thread = threading.Thread(target=self.start_move)
        self.capture_thread = threading.Thread(target=self.capture_images)
        self.move_thread.start()
        self.capture_thread.start()

    def start_capture(self):
        """给出信号开始采集图像"""
        self.frame_count = 0
        self.controller.stop_move = False
        self.move_signal.set()
        self.capture_signal.set()

    def start_move(self):
        self.move_signal.wait()
        self.controller.circular_motion()
    
    def stop_capture(self):
        """停止采集图像"""
        self.capture_signal.clear()
        self.controller.stop_move = True

    def stop(self):
        """结束采集并关闭相机"""
        self.keep_running = False
        self.capture_thread.join()
        self.move_thread.join()
        self.pipeline.stop()
        self.controller.disconnect()

    def capture_images(self):
        """持续采集图像和关节角度，直到停止采集为止。"""
        while self.keep_running:
            self.capture_signal.wait()  

            if self.frame_count >= self.image_len: 
                self.stop_capture()
                #print("Captured 10 images, stopping capture.")
                break

            frames = self.pipeline.wait_for_frames()
            # 获取当前的关节角度
            joint_angles = self.controller.get_current_joints()
            aligned_frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame:
                continue

            # 将图像数据转化为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())


            # 保存RGB图像
            timestamp = int(time.time() * 1000)  
            rgb_filename = os.path.join(self.image_dir, f"frame_{self.frame_count:04}.png")
            cv2.imwrite(rgb_filename, color_image)
            
            if self.pcd_experiments:
                depth_filename = os.path.join(self.depth_dir, f"depth_{self.frame_count:04}.npy")
                np.save(depth_filename, depth_image)

            # 记录关节角度到文件
            ur5_kinematics = UR5Kinematics()
            camera_pose = ur5_kinematics.get_camera_pose(joint_angles)
            rotation_matrix = camera_pose[:3, :3]

            joint_filename = os.path.join(self.joints_dir, f"pose_{self.frame_count:04}.npy")
            np.save(joint_filename, camera_pose)
            self.frame_count += 1

            if self.image_len == 2:
                time.sleep(3)
            else :
                time.sleep(1)

    def clear_directories(self):
        """清空图像和关节角度目录。"""
        for dir_path in [self.image_dir, self.joints_dir]:
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    def disconnect(self):
        """停止相机并断开与机械臂的连接。"""
        self.stop()


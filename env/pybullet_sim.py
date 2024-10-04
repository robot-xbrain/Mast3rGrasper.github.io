import os
import sys
import threading
import pybullet_data
import yaml
import numpy as np
from PIL import Image
import pybullet as pb
from tqdm import trange
from collections import namedtuple
from env.cameras import RealSenseD435i
import cv2
from PIL import Image
import time
import random
from env.oja import Oja
import math
import gc
sys.path.append("/home/desc/jzy/mast3r-grasp-sim/")
from grasper.start_camera.camera_pose import UR5Kinematics
from grasper.utils.grasp_utils import get_joint_kinematics_tradition, get_xyzrpy_kinematics

#arg = yaml.load(open(sys.argv[1], 'r'), yaml.Loader)
arg = yaml.load(open('cfg/pybullet_sim_config.yaml', 'r'), yaml.Loader)
arg = namedtuple('arg', arg.keys())(**arg)

from env.constants import *
from scipy.spatial.transform import Rotation as R

# 设置一个OjaPick类
class OjaPick:
    def __init__(self, image_len, prompt):
        # object spawning sequence
        # 创建了一个名为robot的Oja类的对象。这个对象用于控制仿真中的机器人。构造函数通过arg.time_step参数传递了时间步长的信息。

        self.start_joint= [eval(i) for i in arg.start_joint]
        self.target_joint = [eval(i) for i in arg.target_joint]
        # self.start_joint= arg.start_joint
        # self.target_joint = arg.target_joint
        #self.robot.home(self.start_joint)
        self.ur5_kinematics = UR5Kinematics()
        
        self.image_path = arg.image_path
        self.pose_path = arg.pose_path
        self.frame_count = 0
        self.image_len = image_len
        self.prompt = prompt
        self.table_hight = 0.853

        # if have prompt, object can not be (must in scence)
        if self.prompt is not None:
            arg.obj_in_scene = False
        
        
        # 初始化了两个属性pose_tcp和cnt_obj，并将它们的值都设置为None。这些属性用于存储机器人的TCP姿势和物体的计数。
        self.pose_tcp, self.cnt_obj = None, None
        
        self.target_obj_idx = None
        self.obj_idx_list = []
        self.mesh_list = []
        self.target_obj_load_idx = None
        
        self.running = False
        self.sim_thread = None
        self.init_state = False
        self.start_simulation()
  
    def start_simulation(self):
        """启动仿真并在后台线程中运行"""
        if not self.running:
            self.running = True
            self.sim_thread = threading.Thread(target=self.run_simulation)
            self.sim_thread.start()
            print("仿真已启动")

    def run_simulation(self):
        """后台运行的仿真逻辑"""
        pb.connect(pb.GUI)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(1. / 240.)
        
        self.robot = Oja(config=RealSenseD435i.CONFIG[0], timestep=arg.time_step, robot_xyz=(0,0,0))
        print("已加载机器人模型!")
        # print base_link position and rotation in world_link
        # base_position, base_orientation = pb.getBasePositionAndOrientation(self.robot.robotUid)    
        # print(f"Base link position: {base_position}")
        # print(f"Base link orientation: {base_orientation}")
        # 设置重力
        pb.setGravity(0, 0, -9.81)  # 确保物体会在重力下落到桌子上
    
        # 调用创建桌子的函数
        self.create_table()
        self.init_state = True
        
        while self.running:
            pb.stepSimulation()
            time.sleep(1. / 240.)

    def stop_simulation(self):
        """停止仿真"""
        self.running = False
        if self.sim_thread is not None:
            self.sim_thread.join()
        pb.disconnect()
        print("仿真已停止")
    
    # 析构函数对象被销毁时断开与PyBullet仿真环境的连接，释放资源。
    def __del__(self):
        pb.disconnect()
        
    def run_simulation_time(self, duration):
        """
        运行仿真指定时间（秒）
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            pb.stepSimulation()  # 逐步运行仿真
            time.sleep(1/240)     # 仿真运行的步长，与时间步长一致，1/240 表示 240Hz 的仿真频率
    
    def create_table(self):
        """
        创建桌子及其四条腿，并确保桌子有碰撞检测
        """
        table_position = [0.4, 0, self.table_hight]  # 将桌面沿着 X 轴移动 0.4米
        table_orientation = [0, 0, 0, 1]  # 无旋转
        
        # 创建桌面的形状 (1米 x 1米 x 0.02米)
        table_collision_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.6, 0.8, 0.01])
        table_visual_shape = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.6, 0.8, 0.01], 
                                                  rgbaColor=[0.2, 0.08, 0.07, 1], specularColor=[0, 0, 0])
        
        # 创建桌面刚体，并确保它是静止的
        self.table = pb.createMultiBody(baseCollisionShapeIndex=table_collision_shape,
                                         baseVisualShapeIndex=table_visual_shape,
                                         basePosition=table_position,
                                         baseOrientation=table_orientation)
        
        # 创建桌子的4条腿
        leg_height = self.table_hight  # 桌子腿的高度
        leg_half_width = 0.02  # 桌子腿的宽度
        
        leg_collision_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[leg_half_width, leg_half_width, leg_height / 2])
        leg_visual_shape = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[leg_half_width, leg_half_width, leg_height / 2], rgbaColor=[0.4, 0.2, 0.1, 1])  # 桌腿的颜色

        # 桌腿的四个角的位置（相对于桌子的中心）
        leg_positions = [
            [0.45 + 0.4, 0.45, leg_height / 2],  # 右前 (X 轴移动 0.4m)
            [0.45 + 0.4, -0.45, leg_height / 2],  # 左前 (X 轴移动 0.4m)
            [-0.45 + 0.4, 0.45, leg_height / 2],  # 右后 (X 轴移动 0.4m)
            [-0.45 + 0.4, -0.45, leg_height / 2]  # 左后 (X 轴移动 0.4m)
        ]

        # 创建四条桌腿并将其放置在桌子的四个角
        for leg_position in leg_positions:
            pb.createMultiBody(baseCollisionShapeIndex=leg_collision_shape,
                                baseVisualShapeIndex=leg_visual_shape,
                                basePosition=leg_position,
                                baseOrientation=[0, 0, 0, 1])
    
            
    def select_object(self):
        if arg.obj_in_scene:
            target_obj_idx = random.randint(0, len(LABEL)-1)
            list_length = random.randint(arg.min_obj_num_one_scene, arg.max_obj_num_one_scene)
            obj_idx_list = [random.randint(0, len(LABEL)-1) for _ in range(list_length)]
            if target_obj_idx not in obj_idx_list:
                replace_idx = random.randint(0, list_length - 1)
                obj_idx_list[replace_idx] = target_obj_idx
        else:
            target_obj_idx = random.randint(0, len(LABEL)-1)
            list_length = random.randint(arg.min_obj_num_one_scene, arg.max_obj_num_one_scene)
            obj_idx_list = [random.randint(0, len(LABEL)-1) for _ in range(list_length)]
        
        self.target_obj_idx = target_obj_idx
        self.obj_idx_list = obj_idx_list
        return True
        
    def get_random_pos(self, workspace):
        random_float_x = random.random()
        random_float_y = random.random()
        x_min, x_max, y_min, y_max, _, _ = workspace
        x = x_max - x_min - 0.2
        y = y_max - y_min - 0.2
        pos_x = 0.1 + x_min + x * random_float_x
        pos_y = 0.1 + y_min + y * random_float_y
        return pos_x, pos_y
    
    def add_object(self, workspace):
        for obj_idx in self.obj_idx_list:
            obj_filename = LABEL_DIR_MAP[obj_idx]
            obj_filename = "assets/simplified_objects/"+obj_filename+".urdf"
            pos_x, pos_y = self.get_random_pos(workspace=workspace)
            obj_load_idx = pb.loadURDF(obj_filename, basePosition=[pos_x, pos_y, self.table_hight+0.07]) 
            if obj_idx == self.target_obj_idx:
                self.target_obj_load_idx = obj_load_idx
            self.mesh_list.append(obj_load_idx)
        return True
    
    def generate_objects(self):
        """
        随机生成几个不同形状的物体放置在桌面上方
        """
        # 生成球体、盒子、圆柱体等物体
        for i in range(5):
            shape_type = random.choice([pb.GEOM_SPHERE, pb.GEOM_BOX, pb.GEOM_CYLINDER])
            
            # 随机选择生成物体的形状
            if shape_type == pb.GEOM_SPHERE:
                collision_shape = pb.createCollisionShape(shapeType=pb.GEOM_SPHERE, radius=0.05)
                visual_shape = pb.createVisualShape(shapeType=pb.GEOM_SPHERE, radius=0.05, rgbaColor=[random.random(), random.random(), random.random(), 1])
            elif shape_type == pb.GEOM_BOX:
                half_extents = [random.uniform(0.02, 0.05), random.uniform(0.02, 0.05), random.uniform(0.02, 0.05)]
                collision_shape = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=half_extents)
                visual_shape = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=half_extents, rgbaColor=[random.random(), random.random(), random.random(), 1])
            elif shape_type == pb.GEOM_CYLINDER:
                collision_shape = pb.createCollisionShape(shapeType=pb.GEOM_CYLINDER, radius=0.03, height=0.1)
                visual_shape = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER, radius=0.03, length=0.1, rgbaColor=[random.random(), random.random(), random.random(), 1])

            # 随机生成位置和质量
            position = [random.uniform(0.2, 0.6), random.uniform(-0.2, 0.2), 1.0]
            mass = random.uniform(0.1, 1.0)
            
            # 创建刚体，并将其放置在桌子上方
            pb.createMultiBody(baseMass=mass,
                                baseCollisionShapeIndex=collision_shape,
                                baseVisualShapeIndex=visual_shape,
                                basePosition=position)
            
        print("已生成多个物体!")

    def remove_object(self):
        for obj_load_idx in self.mesh_list:
            pb.removeBody(obj_load_idx)
        self.mesh_list.clear()
        gc.collect()
        return True
    
    def get_prompt(self):
        if self.prompt is None:
            prompt = LABEL[self.target_obj_idx]
        else:
            prompt = self.prompt
        return prompt
    
    def get_quat_and_t(self, pose):
        rotation = pose[:3, :3]
        t = pose[:3, -1]
        r = R.from_matrix(rotation)
        quat = r.as_quat()
        rpy = r.as_euler('xyz', degrees=True)
        return quat, t
    
    def camera_vis(self, position, orientation):
        camera_visual = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[0, 1, 0, 1])

        # 定义相机的位姿 (translation和rotation)
        # 假设你有相机的位姿(平移和旋转), 例如:
        camera_position = position  # 相机的平移 (x, y, z)
        camera_orientation = orientation  # 相机的旋转 (四元数形式)

        # 在PyBullet中将相机的位置和旋转可视化
        camera_body = pb.createMultiBody(
            baseVisualShapeIndex=camera_visual,
            basePosition=camera_position,
            baseOrientation=camera_orientation
        )
    
    def camera_pose_adjust(self, joint):
        robot_pose = get_xyzrpy_kinematics(joint)
        robot_pose = robot_pose.A
        print("robot_pose: ", robot_pose, robot_pose.shape, robot_pose.dtype)
        #robot_pose[2, 3] -= 0.1
        robot_pose[3, 2] -= 0.1

        rotation = robot_pose[:3, :3]
        translation = robot_pose[:3, -1]
        r = R.from_matrix(rotation)
        rpy_angles = r.as_euler('xyz', degrees=False)  # 'xyz'表示roll, pitch, yaw顺序

        print("RPY角度: ", rpy_angles)
        #self.robot.set_tcp_pose(translation,rpy_angles)
        # joint = get_joint_kinematics_tradition(xyz=translation, rotation_matrix=rotation)
        # robot_pose = get_xyzrpy_kinematics(joint.q)
        current_joints = self.robot.get_joint()
        print("robot_pose: ", robot_pose)
        print("current_joints: ", current_joints)
        # print("joint: ", joint.q)
        return current_joints

    def caputer_image(self):
        time.sleep(2)
        self.frame_count = 0
        if self.image_len == 2:
            self.robot._set_joint(idx=IDX_ARM, pos=self.start_joint)
            pose = self.ur5_kinematics.get_camera_pose_sim(joint_angles=self.start_joint)
            pose_path = os.path.join(self.pose_path, f"pose_{self.frame_count:04}.npy")
            np.save(pose_path, pose)
            quat, t = self.get_quat_and_t(pose=pose)
            target_obj_pos, _ = pb.getBasePositionAndOrientation(self.target_obj_load_idx)
            color, depth, _, pixel_x1, pixel_y1 = self.robot.get_image(rotation=quat, position=t, target_obj_pos=target_obj_pos)
            image_path = os.path.join(self.image_path, f"frame_{self.frame_count:04}.png")
            cv2.imwrite(image_path, color)
            self.frame_count += 1

            self.robot._set_joint(idx=IDX_ARM, pos=self.target_joint)
            pose = self.ur5_kinematics.get_camera_pose_sim(joint_angles=self.target_joint)
            pose_path = os.path.join(self.pose_path, f"pose_{self.frame_count:04}.npy")
            np.save(pose_path, pose)
            quat, t = self.get_quat_and_t(pose=pose)
            target_obj_pos, _ = pb.getBasePositionAndOrientation(self.target_obj_load_idx)
            color, depth, _, pixel_x2, pixel_y2 = self.robot.get_image(rotation=quat, position=t, target_obj_pos=target_obj_pos)
            image_path = os.path.join(self.image_path, f"frame_{self.frame_count:04}.png")
            cv2.imwrite(image_path, color)
            return pixel_x1, pixel_y1

        if arg.capture_type == "time":
            while True:
                if self.frame_count >= self.image_len:
                    break
                
                joint = self.robot.get_joint()
                pose = self.ur5_kinematics.get_camera_pose_sim(joint_angles=joint)
                pose_path = os.path.join(self.pose_path, f"pose_{self.frame_count:04}.npy")
                np.save(pose_path, pose)
                quat, t = self.get_quat_and_t(pose=pose)
                color, depth, _ = self.robot.get_image(rotation=quat, position=t)
                image_path = os.path.join(self.image_path, f"frame_{self.frame_count:04}.png")
                cv2.imwrite(image_path, color)
                self.frame_count += 1
                time.sleep(1)
        else:
            txt_file = "./grasper/start_camera/straight_line_joints.txt"
            joint_data = []
            with open(txt_file, 'r') as file:
                for line in file:
                    joint_angles = [float(value) for value in line.split()]
                    joint_data.append(joint_angles)
        
            for joints in joint_data:
                self.robot._set_joint(idx=IDX_ARM, pos=joints)
                pose = self.ur5_kinematics.get_camera_pose_sim(joint_angles=joints)
                pose_path = os.path.join(self.pose_path, f"pose_{self.frame_count:04}.npy")
                np.save(pose_path, pose)
                quat, t = self.get_quat_and_t(pose=pose)
                color, depth, _ = self.robot.get_image(rotation=quat, position=t)
                image_path = os.path.join(self.image_path, f"frame_{self.frame_count:04}.png")
                cv2.imwrite(image_path, color)
                self.frame_count += 1

    def env_sim(self):
        while True:
            if self.init_state ==True:
                break
            time.sleep(0.1)
        if self.select_object():
            idx_select_word = random.randint(0, len(LANG_TEMPLATES)-1)
            prompt = self.get_prompt()
            print(f"select object success, {LANG_TEMPLATES[idx_select_word].format(keyword=prompt)}")
        else:
            print("select failed, exit")
            self.__del__()
            exit()
        if self.add_object(workspace=WORKSPACE_LIMITS):
            print("objects have been loaded in the scene")
        else:
            print("load objects failed, exit")
            self.__del__()
            exit()
        time.sleep(1)
        pixel_x, pixel_y =self.caputer_image()
        return pixel_x, pixel_y
    
    def grasp_sim(self, rotation, translation, width):
        joints = get_joint_kinematics_tradition(xyz=translation, rotation_matrix=rotation)
        print("joints: ", joints)
        grasper_pose = get_xyzrpy_kinematics(joints.q)
        print("grasper_pose: ", grasper_pose)

        self.robot._set_joint(idx=IDX_ARM, pos=joints.q)
        self.robot.set_gripper_joint(pos=width)

        time.sleep(10)
        if self.remove_object():
            print("objects in workspace have been removed")
        else:
            print("objects remove failed, exit")
            self.__del__()
            exit()
        return 

    
# test 
if __name__ == "__main__":
    env_sim = OjaPick(image_len=2, prompt=None)
    env_sim.env_sim()
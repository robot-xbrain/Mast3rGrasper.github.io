import pybullet as pyb
import numpy as np

PI = 3.14159265359
ARM_LOWER = [-2*PI, -2*PI, -2*PI, -2*PI, -2*PI, -2*PI]
ARM_UPPER = [2*PI, 2*PI, 2*PI, 2*PI, 2*PI, 2*PI]
ARM_RANGE = [4*PI, 4*PI, 4*PI, 4*PI, 4*PI, 4*PI]
ARM_HOME = [PI/4, -PI/2, -PI/2, -PI/2, PI/2, 0]
ARM_DAMP = [0.01, 0.01, 0.001, 0.001, 0.001, 0.001]

GRIPPER_LOWER = [0, -0.8, 0, 0, -0.8, 0]
GRIPPER_UPPER = [0.8, 0, 0.8, 0.8, 0, 0.8]
GRIPPER_RANGE = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
GRIPPER_HOME = [0, 0, 0, 0, 0, 0]
GRIPPER_DAMP = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

IDX_ARM = [1, 2, 3, 4, 5, 6]
IDX_GRIPPER = [10, 12, 14, 15, 17, 19]
IDX_TCP = 20
IDX_CAMERA = 23
IDX_CAMERA2 = 26

class Oja:
    def __init__(self, config, timestep=1/240, robot_xyz=(0, 0, 0), robot_rpy=(0, 0, 0)):
        self.robotUid = None
        self.timestep = timestep
        self.config = config
        self.load_robot(robot_xyz, robot_rpy)
        self.home()
    def load_robot(self, robot_xyz=(0, 0, 0), robot_rpy=(0, 0, 0), joint_info=True):
        self.robotUid = pyb.loadURDF("./assets/ur5/oja/oja5.urdf.xacro", robot_xyz,
                                     pyb.getQuaternionFromEuler(robot_rpy), useFixedBase=True)
                                     
        if joint_info:
            print('================================== Joint Info ==================================')
            for i in range(pyb.getNumJoints(self.robotUid)):
                info = pyb.getJointInfo(self.robotUid, i)
                if info[2] == 0:
                    jtype = 'revolute'
                elif info[2] == 1:
                    jtype = 'prismatic'
                elif info[2] == 2:
                    jtype = 'spherical'
                elif info[2] == 3:
                    jtype = 'planner'
                elif info[2] == 4:
                    jtype = 'fixed'
                else:
                    jtype = 'unknown'
                print('{:02d}'.format(info[0]), '  ', jtype, '\t', info[1].decode("utf-8"), ' ', info[12].decode("utf-8"))
            print('================================================================================')

    def home(self,position = ARM_HOME):
        self._set_joint(IDX_ARM, position, True)
        self._set_joint(IDX_GRIPPER, GRIPPER_HOME, True)

    def get_tcp_pose(self):
        return pyb.getLinkState(self.robotUid, IDX_TCP, computeForwardKinematics=1)[4:6]

    def set_gripper_joint(self, pos, reset=False):
        self._set_joint(IDX_GRIPPER, [pos, -pos, pos, pos, -pos, pos], reset)
        
    def control_gripper_width(self, width):          
        pos = 0.85-width
        self.set_gripper_joint(pos=pos)

    def set_tcp_pose(self, pos, rot, reset=True):
        for _ in range(10): 
            tcp_pos, tcp_rot = self.get_tcp_pose()
            if (np.abs(np.array(tcp_pos) - pos).sum() > 0.001) or \
                    (np.abs(np.array(tcp_rot) - rot).sum() > 0.005):
                jpos = pyb.calculateInverseKinematics(self.robotUid, IDX_TCP, pos, rot,
                                                      ARM_LOWER+GRIPPER_LOWER, ARM_UPPER+GRIPPER_UPPER,
                                                      ARM_RANGE+GRIPPER_RANGE, ARM_HOME+GRIPPER_HOME,
                                                      ARM_DAMP+GRIPPER_DAMP, pyb.IK_SDLS)
                jpos = jpos[:6]
                self._set_joint(IDX_ARM, jpos, reset)
            else:
                return True
        # return False
        return True
    
    def get_joint(self):
        joint_states = [pyb.getJointState(self.robotUid, jointIndex) for jointIndex in range(pyb.getNumJoints(self.robotUid))]
            
        # 从 joint_states 中选择前六个关节的角度信息
        #joint_angles = [joint_state[0] for joint_state in joint_states[:6]]
        joint_angles = [joint_state[0] for joint_state in joint_states[1:7]]
        return joint_angles

    def apply_speed_tcp(self, vel_xyz, vel_rpy, pose_tcp,
                        relative=False, reset=False, timestep=None):
        timestep = timestep if timestep is not None else np.random.uniform(self.timestep[0], self.timestep[1])
        if len(pose_tcp[1]) == 3:
            euler = True
        else:
            euler = False
        if relative:
            tcp_r = pyb.getQuaternionFromEuler(pose_tcp[1]) if euler else pose_tcp[1]
            pos_goal = [timestep * vel_xyz[i] for i in range(3)]
            rot_goal = [timestep * vel_rpy[i] for i in range(3)]
            pos, rot = pyb.multiplyTransforms(pose_tcp[0], tcp_r,
                                              pos_goal, pyb.getQuaternionFromEuler(rot_goal))
            self.set_tcp_pose(pos, rot, reset)
        else:
            tcp_r = pyb.getEulerFromQuaternion(pose_tcp[1]) if not euler else pose_tcp[1]
            pos = [pose_tcp[0][i] + timestep * vel_xyz[i] for i in range(3)]
            rot = [tcp_r[i] + timestep * vel_rpy[i] for i in range(3)]
            rot = pyb.getQuaternionFromEuler(rot)
            self.set_tcp_pose(pos, rot, reset)
        return pos, \
               pyb.getEulerFromQuaternion(rot) if euler else rot

    def _set_joint(self, idx, pos, reset=False):
        if reset:
            pyb.resetJointStatesMultiDof(self.robotUid, idx, [[i] for i in pos], [[0] for _ in range(len(pos))])
        else:
            pyb.setJointMotorControlMultiDofArray(self.robotUid, idx, pyb.POSITION_CONTROL, [[i] for i in pos])


    def tf(self,x,y,z):
        # 假设物体相对于相机的位置
        object_in_camera_coord = [x, y, z]  # 三维坐标

        # 获取相机的位置和朝向（相对于机器人基座）
        camera_pos, camera_quat = pyb.getLinkState(self.robotUid, IDX_CAMERA)[:2]

        # 创建相机的变换矩阵（相对于机器人基座）
        camera_transform_matrix = pyb.getMatrixFromQuaternion(camera_quat)
        camera_transform_matrix = np.array(camera_transform_matrix).reshape(3, 3)
        camera_transform_matrix = np.vstack((camera_transform_matrix, [0, 0, 0]))
        # 将camera_pos扩展为4个元素
        camera_pos = np.append(camera_pos, 1.0)
        camera_transform_matrix = np.hstack((camera_transform_matrix, np.array(camera_pos).reshape(4, 1)))

        # 将物体的坐标从相机坐标系转换到机器人基座坐标系
        object_in_robot_coord = np.dot(camera_transform_matrix, [object_in_camera_coord[0], object_in_camera_coord[1], object_in_camera_coord[2], 1])

        # 提取物体在机器人基座坐标系中的坐标
        object_x = object_in_robot_coord[0]
        object_y = object_in_robot_coord[1]
        object_z = object_in_robot_coord[2]
        return object_x,object_y,object_z


    def get_image(self, rotation, position, target_obj_pos):

        # OpenGL camera settings.
        config = self.config
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pyb.getMatrixFromQuaternion(rotation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        position[2] += 0.121503 + 0.853
        lookat = position + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = pyb.computeViewMatrix(position, lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi
        
        # 使用PyBullet调试线渲染相机视线
        pyb.addUserDebugLine(position, lookat, lineColorRGB=[0, 1, 0], lineWidth=2)

        # 使用PyBullet调试文本标记相机的位置
        pyb.addUserDebugText("Camera", position, textColorRGB=[1, 0, 0], textSize=1.5)

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = pyb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pyb.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags=pyb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pyb.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)
        pixel_x, pixel_y = self.world_to_image_coordinates(target_obj_pos, viewm, projm, config["image_size"])

        return color, depth, segm, pixel_x, pixel_y


    def world_to_image_coordinates(self, pos, view_matrix, projection_matrix, image_size):
        # 将view_matrix和projection_matrix转换为numpy数组
        view_matrix = np.array(view_matrix).reshape(4, 4).T
        projection_matrix = np.array(projection_matrix).reshape(4, 4).T

        pos_hom = np.append(pos, 1.0)

        pos_cam = np.dot(view_matrix, pos_hom)

        pos_clip = np.dot(projection_matrix, pos_cam)

        pos_ndc = pos_clip[:3] / pos_clip[3]

        pixel_x = int((pos_ndc[0] * 0.5 + 0.5) * image_size[1])  # 图像宽度
        pixel_y = int((1 - (pos_ndc[1] * 0.5 + 0.5)) * image_size[0])  # 图像高度，y轴翻转

        return pixel_x, pixel_y

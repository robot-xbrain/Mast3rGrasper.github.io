import pyrealsense2 as rs
import numpy as np
import cv2
import time
import rtde_control
import rtde_receive
import threading

class UR5Controller:
    def __init__(self, robot_host):
        self.robot_host = robot_host
        self.rtde_c = rtde_control.RTDEControlInterface(robot_host)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_host)
        self.stop_move = False

    def get_current_joints(self):
        """获取当前的关节角度。"""
        return self.rtde_r.getActualQ()

    def move_to_joints(self, target_joints, speed=0.3, acceleration=0.3):
        """
        使机器人移动到给定的关节角度。
        
        :param target_joints: 目标关节角度的列表或数组。
        :param speed: 运动速度，默认值为0.3。
        :param acceleration: 运动加速度，默认值为0.3。
        """
        self.rtde_c.moveJ(target_joints, speed, acceleration)
        while True:
            current_joints = self.get_current_joints()
            if all(abs(c - t) < 0.01 for c, t in zip(current_joints, target_joints)):
                break
            time.sleep(0.01)

    def disconnect(self):
        self.rtde_c.disconnect()

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)

    return pipeline

def start_video_recording(pipeline):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter('robot_movement.mp4', fourcc, 30.0, (1280, 720))  
    return out

def stop_realsense_recording(pipeline, out):
    out.release() 
    pipeline.stop()  
    cv2.destroyAllWindows()

def record_video(pipeline, out, start_recording_event, stop_event):
    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow('RealSense', color_image)

            if start_recording_event.is_set():
                out.write(color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    robot_host = "192.168.1.161"
    ur5 = UR5Controller(robot_host)

    start_joint = [-29.94 * np.pi / 180, -86.55 * np.pi / 180, 68.86 * np.pi / 180, -67 * np.pi / 180, -76.5 * np.pi / 180, 161 * np.pi / 180]
    target_joint = [25.67 * np.pi / 180, -81.52 * np.pi / 180, 64.83 * np.pi / 180, -64.37 * np.pi / 180, -110.02 * np.pi / 180, 198 * np.pi / 180]

    pipeline = start_realsense()

    start_recording_event = threading.Event()
    stop_event = threading.Event()

    out = start_video_recording(pipeline)
    video_thread = threading.Thread(target=record_video, args=(pipeline, out, start_recording_event, stop_event))
    video_thread.start()

    ur5.move_to_joints(start_joint)

    start_recording_event.set()

    ur5.move_to_joints(target_joint)

    stop_event.set()
    video_thread.join()

    stop_realsense_recording(pipeline, out)

    ur5.disconnect()

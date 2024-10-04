#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose
import math


class MoveRobot():
    def __init__(self):
        self.robot = moveit_commander.robot.RobotCommander()
        self.arm_group = moveit_commander.move_group.MoveGroupCommander("manipulator")

        self.arm_group.set_max_acceleration_scaling_factor(0.1)
        self.arm_group.set_max_velocity_scaling_factor(0.1)

        self.target_pose_sub = rospy.Subscriber("/end_effector_target_pose", PoseStamped, self.target_pose_callback)

    def target_pose_callback(self, msg):
        """
        回调函数：接收目标位姿并让机械臂移动到该位姿
        """
        rospy.loginfo("Received target pose: {}".format(msg.pose))

        self.arm_group.set_pose_target(msg.pose)

        success = self.arm_group.go(wait=True)
        if success:
            rospy.loginfo("Move to target pose succeeded")
        else:
            rospy.logwarn("Move to target pose failed")

        self.arm_group.clear_pose_targets()
    
    def gripperOpen(self):
        self.req.cmd = str('0')
        self.gripper_client.call(self.req)

    def gripperClose(self):
        self.req.cmd = str('145')
        self.gripper_client.call(self.req)

    def stop(self):
        moveit_commander.roscpp_initializer.roscpp_shutdown()
  


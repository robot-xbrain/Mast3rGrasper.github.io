#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped

def publish_target_pose(position, orientation, frame_id="base_link"):
    """
    发布机械臂末端执行器的目标位姿

    :param position: 目标位置，格式为[x, y, z]
    :param orientation: 目标方向，格式为四元数[x, y, z, w]
    :param frame_id: 参考坐标系，默认为"base_link"
    """
    pub = rospy.Publisher('/end_effector_target_pose', PoseStamped, queue_size=10)
    
    target_pose = PoseStamped()
    target_pose.header.stamp = rospy.Time.now()
    target_pose.header.frame_id = frame_id
    
    target_pose.pose.position.x = position[0]
    target_pose.pose.position.y = position[1]
    target_pose.pose.position.z = position[2]

    target_pose.pose.orientation.x = orientation[0]
    target_pose.pose.orientation.y = orientation[1]
    target_pose.pose.orientation.z = orientation[2]
    target_pose.pose.orientation.w = orientation[3]

    pub.publish(target_pose)

    rospy.loginfo("Published Target Pose: Position (%f, %f, %f) Orientation (%f, %f, %f, %f)",
                  target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z,
                  target_pose.pose.orientation.x, target_pose.pose.orientation.y, 
                  target_pose.pose.orientation.z, target_pose.pose.orientation.w)



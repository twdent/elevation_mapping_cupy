#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

def publish_camera_tf():
    rospy.init_node('camera_tf_publisher')
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    rate = rospy.Rate(1)  

    while not rospy.is_shutdown():
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "v4l2_camera_parent"
        transform.child_frame_id = "wide_angle_camera_rear_camera_parent"
        transform.transform.translation.x = camera_x
        transform.transform.translation.y = camera_y
        transform.transform.translation.z = camera_z
        transform.transform.rotation.x = camera_quaternion_x
        transform.transform.rotation.y = camera_quaternion_y
        transform.transform.rotation.z = camera_quaternion_z
        transform.transform.rotation.w = camera_quaternion_w

        tf_broadcaster.sendTransform(transform)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_camera_tf()
    except rospy.ROSInterruptException:
        pass

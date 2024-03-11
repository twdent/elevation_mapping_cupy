#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

rospy.init_node('image_publisher', anonymous=True)
r_channel_pub = rospy.Publisher('/traversable_segmentation/red_channel', Image, queue_size=10)
g_channel_pub = rospy.Publisher('/traversable_segmentation/green_channel', Image, queue_size=10)
b_channel_pub = rospy.Publisher('/traversable_segmentation/blue_channel', Image, queue_size=10)


def image_callback(msg):
    bridge = CvBridge()
    
    # Convert ROS Image message to OpenCV image
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # Normalize and convert to float16
    normalized_image = cv_image.astype(np.float32) / 255.0

    # Split RGB image to three mono images
    r_channel = normalized_image[:, :, 0]  # Red channel
    g_channel = normalized_image[:, :, 1]  # Green channel
    b_channel = normalized_image[:, :, 2]  # Blue channel

    # Create Image messages for R, G, B channels
    r_msg = bridge.cv2_to_imgmsg(r_channel, encoding='32FC1')
    g_msg = bridge.cv2_to_imgmsg(g_channel, encoding='32FC1')
    b_msg = bridge.cv2_to_imgmsg(b_channel, encoding='32FC1')

    # Publish R, G, B channels to new topics
    r_channel_pub.publish(r_msg)
    g_channel_pub.publish(g_msg)
    b_channel_pub.publish(b_msg)
    # print("working")



rospy.Subscriber('/traversable_segmentation/class_probabilities', Image, image_callback)
rospy.spin()

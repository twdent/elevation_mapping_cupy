#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import numpy as np

import PIL
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo,Image


class DummyImageNode:
    '''Custom ROS node that publishes a dummy image as if it we received from the wide angle camera.'''
    def __init__(self):
        rospy.init_node('dummy_img_node', anonymous=False)

        # Subscribers
        self.image_sub = rospy.Subscriber('/wide_angle_camera_rear/image_color_rect/compressed', CompressedImage, self.image_callback)
        self.info_sub = rospy.Subscriber('/wide_angle_camera_rear/camera_info', CameraInfo, self.info_callback)

        # Publisher
        self.segmented_image_pub = rospy.Publisher('/dummy_img_topic', Image, queue_size=10)
        self.segmented_info_pub = rospy.Publisher('/dummy_img_topic/camera_info', CameraInfo, queue_size=10)
        # Initialize the CvBridge
        self.bridge = CvBridge()

        self.count = 0

      

    def image_callback(self, msg):

        

        try:
            # Convert compressed image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            height, width, channels = cv_image.shape
            # Create a dummy image
            dummy_image = np.zeros(cv_image.shape, dtype=np.uint8)
            dummy_image[:, self.count:, 1] = 255
            self.count += 1
            if self.count == width:
                self.count = 0

            # Convert the dummy image back to Image format
            dummy_msg = self.bridge.cv2_to_imgmsg(dummy_image, encoding='passthrough')
            # dummy_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')

            #set the frame id
            dummy_msg.header.frame_id = msg.header.frame_id
            #set the timestamp
            dummy_msg.header.stamp = rospy.Time.now()

            # Publish the dummy image
            self.segmented_image_pub.publish(dummy_msg)

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
    
    def info_callback(self, msg):
        # Publish the camera info
        self.segmented_info_pub.publish(msg)


if __name__ == '__main__':
    #test the segmentation node

    try:
        node = DummyImageNode()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
#!/home/twdenton/anaconda3/envs/thesis/bin/python

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo,Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import PIL

from pytictac import Timer, CpuTimer
import time

# Import Segformer model here
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch import nn
import torch
import tf
import os, sys

rel_path = '../../config/core/segformer-b0-finetuned-HikingHD'
MODEL_PATH = os.path.join(os.path.dirname(__file__), rel_path)


class SegmentationNode:
    def __init__(self):
        rospy.init_node('segmentation_node', anonymous=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.listener = tf.TransformListener()

        # Publisher
        self.segmented_image_pub = rospy.Publisher('/traversable_segmentation/segmentation/compressed', CompressedImage, queue_size=1)
        self.class_probabilities_pub = rospy.Publisher('/traversable_segmentation/class_probabilities', Image, queue_size=1)
        self.segmented_info_pub = rospy.Publisher('/traversable_segmentation/camera_info', CameraInfo, queue_size=1)
        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Initialize model here
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH, device_map = self.device)

        # # Subscribers
        self.image_sub = rospy.Subscriber('/wide_angle_camera_rear/image_color_rect/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        self.info_sub = rospy.Subscriber('/wide_angle_camera_rear/camera_info', CameraInfo, self.info_callback, queue_size=1)
        #Subscribers hdr camera //v4l2_camera/image_raw_throttle/compressed
        # self.image_sub = rospy.Subscriber('/v4l2_camera/image_raw_throttle/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        # self.info_sub = rospy.Subscriber('/v4l2_camera/camera_info_throttle', CameraInfo, self.info_callback, queue_size=1)
        # fixed topic /hdr_camera/image_raw/compressed
        # self.image_sub = rospy.Subscriber('/hdr_camera/image_raw/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        # self.info_sub = rospy.Subscriber('/hdr_camera/camera_info', CameraInfo, self.info_callback, queue_size=1)

    def image_callback(self, msg):
        with Timer('Segmentation Callback'):
            start_time = time.time()
            # Create CUDA events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # Synchronize the GPU to ensure accurate timing
            torch.cuda.synchronize()
            # Record the start time
            start_event.record()  

            #fix the frame id if it is the hdr camera
            if msg.header.frame_id == '':
                msg.header.frame_id = 'hdr_cam'          
            
            # Convert compressed image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Color conversion
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # Perform segmentation model
            segmented_image, class_probabilities = self.segment_image(cv_image)

            # Hot fix t okeep class_probs with 3 channels until multi-channel segmentation is implemented
            class_probabilities = np.concatenate((class_probabilities, np.zeros((class_probabilities.shape[0], class_probabilities.shape[1], 1))), axis=2)

            # Convert the segmented image back to Image format
            # segmented_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding='passthrough')

            # Convert the segmented image back to CompressedImage format (used for testing)
            # segmented_msg = self.bridge.cv2_to_compressed_imgmsg(segmented_image) 

            # test for displaying class probs in category channels with unit using seg publisher
            segmented_image = (class_probabilities * 255).astype(np.uint8)
            # hack to convert blue - green colours to red - yellow (in BGR)
            red_yellow = np.zeros_like(segmented_image)
            red_yellow[:,:,2] = segmented_image[:,:,0] + segmented_image[:,:,1]
            red_yellow[:,:,1] = segmented_image[:,:,1]

            segmented_msg = self.bridge.cv2_to_compressed_imgmsg(red_yellow)

            #set the frame id
            segmented_msg.header.frame_id = msg.header.frame_id
            #set the timestamp
            segmented_msg.header.stamp = msg.header.stamp

            # Convert class_probabilities to an Image message
            class_probabilities_msg = self.bridge.cv2_to_imgmsg(class_probabilities, encoding='passthrough')
            # Convert the class_probabilities to a CompressedImage message
            # class_probabilities_msg = self.bridge.cv2_to_compressed_imgmsg(class_probabilities)
            # class_probabilities_msg = self.create_probability_image(class_probabilities)
            # Set the frame ID and timestamp
            class_probabilities_msg.header.frame_id = msg.header.frame_id
            class_probabilities_msg.header.stamp = msg.header.stamp
            
            # Publish the segmented image and class probabilities
            self.segmented_image_pub.publish(segmented_msg)
            self.class_probabilities_pub.publish(class_probabilities_msg)

            end_time = time.time()
            # Record the end time
            end_event.record()

            # Synchronize the GPU again to ensure accurate timing
            torch.cuda.synchronize()

            # Calculate the time elapsed
            elapsed_time_ms = start_event.elapsed_time(end_event)
            # rospy.loginfo(f"Segmentation processing time: {end_time - start_time} seconds")
            # rospy.loginfo(f"CUDA seg processing time    : {elapsed_time_ms/1000} seconds")


    def create_probability_image(self, class_probabilities):
        # Normalize probabilities to range [0, 255] and convert to uint8
        probabilities_uint8 = (class_probabilities * 255).astype(np.uint8)

        # Create an Image message
        probability_image_msg = Image()
        probability_image_msg.height = class_probabilities.shape[0]
        probability_image_msg.width = class_probabilities.shape[1]
        probability_image_msg.encoding = 'mono8'
        probability_image_msg.step = probability_image_msg.width
        probability_image_msg.data = probabilities_uint8.tobytes()

        return probability_image_msg
    
    def info_callback(self, msg):

        # Publish the camera info
        self.segmented_info_pub.publish(msg)
        
        # info sub not recorded properly, using fixed static transform from wide angle camera
        # static_transform_publisher x y z qx qy qz qw frame_id child_frame_id  period_in_ms
        # static_transform_publisher -0.00451632 -0.09041891 0.04183124 -0.00371707 0.10704935 0.99422573 0.00646622 wide_angle_camera_rear_camera_parent hdr_cam
        # self.listener.waitForTransform('wide_angle_camera_rear_camera_parent', 'hdr_cam', msg.header.stamp, rospy.Duration(1.0))
   
            

    def segment_image(self, image):
        # Perform segmentation 
        #conver to PIL image on device
        image = PIL.Image.fromarray(image)

        # Preprocess image
        # with Timer('Preprocessing time'):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        # with Timer('Inference time'):    
        outputs = self.model(**inputs)

        logits = outputs.logits # shape (batch_size, num_labels, height/4, width/4)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode='bilinear',
            align_corners=False
        )

        # Convert to probabilities for each class
        upsampled_probs = nn.functional.softmax(upsampled_logits, dim=1)
        # as a numpy array
        upsampled_probs = upsampled_probs[0].cpu().detach().numpy().transpose(1, 2, 0)

        # Second, apply argmax on the class dimension
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu()
        #ovelay the segmentation on the original image
        seg_overlay = self.get_seg_overlay(image, pred_seg)

        return seg_overlay, upsampled_probs
        # #logit array return
        # upsampled_logits = upsampled_logits[0].detach().numpy().transpose(1, 2, 0)
        # return upsampled_logits
    
    def get_seg_overlay(self, image, seg):
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        palette = np.array([
                            [255, 0, 0],      # traversable
                            [255, 255, 0],      # untraversable
                            [0, 0, 0],          # unlabeled
                            ])
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        # # Show image + mask
        # img = np.array(image) * 0.7 + color_seg * 0.3
        img = color_seg
        img = img.astype(np.uint8)

        return img

if __name__ == '__main__':
    try:
        segmentation_node = SegmentationNode()
        print("Segmentation node started")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

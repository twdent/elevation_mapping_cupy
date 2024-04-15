#!/home/twdenton/anaconda3/envs/thesis/bin/python

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo,Image
import message_filters
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

        self.palette = torch.tensor([
                            [255, 0, 0],      # traversable
                            [255, 255, 0],      # untraversable
                            [0, 0, 0],          # unlabeled
                            ], 
                            dtype=torch.uint8,
                            device=self.device)

        # Publishers
        self.segmented_image_pub = rospy.Publisher('/traversable_segmentation/seg_img/compressed', CompressedImage, queue_size=1)
        
        self.class_probs_rear_pub = rospy.Publisher('/traversable_segmentation/class_probabilities_rear', Image, queue_size=1)
        self.class_probs_front_pub = rospy.Publisher('/traversable_segmentation/class_probabilities_front', Image, queue_size=1)
        self.class_probs_hdr_pub = rospy.Publisher('/traversable_segmentation/class_probabilities_hdr', Image, queue_size=1)
        
        self.segmented_info_rear_pub = rospy.Publisher('/traversable_segmentation/camera_info_rear', CameraInfo, queue_size=1)
        self.segmented_info_front_pub = rospy.Publisher('/traversable_segmentation/camera_info_front', CameraInfo, queue_size=1)
        self.segmented_info_hdr_pub = rospy.Publisher('/traversable_segmentation/camera_info_hdr', CameraInfo, queue_size=1)
        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Initialize model here
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH, device_map = self.device)
        self.model.eval()

        # # Subscribers
        self.image_sub_rear = message_filters.Subscriber('/wide_angle_camera_rear/image_color_rect/compressed', CompressedImage, queue_size=1, buff_size=2**24)
        self.image_sub_front = message_filters.Subscriber('/wide_angle_camera_front/image_color_rect/compressed', CompressedImage, queue_size=1, buff_size=2**24)
        # self.image_sub_hdr = message_filters.Subscriber('/hdr_camera/image_raw/compressed', CompressedImage, queue_size=1, buff_size=2**24)

        print("Segmentation waiting for camera info message...")
        self.camera_info_msg_rear = rospy.wait_for_message('/wide_angle_camera_rear/camera_info', CameraInfo)
        self.camera_info_msg_front = rospy.wait_for_message('/wide_angle_camera_front/camera_info', CameraInfo)
        # self.camera_info_msg_hdr = rospy.wait_for_message('/hdr_camera/camera_info', CameraInfo)
        
        self.orig_height, self.orig_width = self.camera_info_msg_rear.height, self.camera_info_msg_rear.width
        new_height, new_width =  128, 128
        
        self.camera_info_msg_rear = self.intrinsic_matrix_rescale(self.camera_info_msg_rear, 
                                                                    self.camera_info_msg_rear.height, self.camera_info_msg_rear.width, 
                                                                    new_height, new_width)
        self.camera_info_msg_front = self.intrinsic_matrix_rescale(self.camera_info_msg_front,
                                                                    self.camera_info_msg_front.height, self.camera_info_msg_front.width,
                                                                    new_height, new_width)
        # self.camera_info_msg_hdr = self.intrinsic_matrix_rescale(self.camera_info_msg_hdr,
        #                                                             self.camera_info_msg_hdr.height, self.camera_info_msg_hdr.width,
        #                                                             new_height, new_width)  

        # ts = message_filters.ApproximateTimeSynchronizer([self.image_sub_rear, self.image_sub_front, self.image_sub_hdr], 1, 0.1)
        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub_rear, self.image_sub_front], 1, 0.1)
        ts.registerCallback(self.image_callback)

    def intrinsic_matrix_rescale(self, camera_info_msg, orig_height, orig_width, new_height, new_width):
        height_scale = new_height / orig_height
        width_scale = new_width / orig_width

        camera_info_msg.K = list(camera_info_msg.K)
        camera_info_msg.K[0] *= width_scale
        camera_info_msg.K[4] *= height_scale 
        camera_info_msg.K[2] *= width_scale
        camera_info_msg.K[5] *= height_scale
        camera_info_msg.K = tuple(camera_info_msg.K) 

        camera_info_msg.P = list(camera_info_msg.P)
        camera_info_msg.P[0] *= width_scale
        camera_info_msg.P[5] *= height_scale 
        camera_info_msg.P[2] *= width_scale
        camera_info_msg.P[6] *= height_scale 
        camera_info_msg.P = tuple(camera_info_msg.P)

        camera_info_msg.height = new_height
        camera_info_msg.width = new_width

        return camera_info_msg
        
    @torch.no_grad()
    def image_callback(self, msg_rear, msg_front):#, msg_hdr):
        with Timer('Segmentation Callback'):

            # Convert compressed image to OpenCV format
            cv_image_rear = self.bridge.compressed_imgmsg_to_cv2(msg_rear, desired_encoding='passthrough')
            cv_image_front = self.bridge.compressed_imgmsg_to_cv2(msg_front, desired_encoding='passthrough')
            # cv_image_hdr = self.bridge.compressed_imgmsg_to_cv2(msg_hdr, desired_encoding='passthrough')

            # Color convert and batch images
            cv_image_rear = cv2.cvtColor(cv_image_rear, cv2.COLOR_BGR2RGB)
            cv_image_front = cv2.cvtColor(cv_image_front, cv2.COLOR_BGR2RGB)
            # cv_image_hdr = cv2.cvtColor(cv_image_hdr, cv2.COLOR_BGR2RGB)

            # Resize images
            cv_image_rear = cv2.resize(cv_image_rear, (self.orig_width, self.orig_height))
            cv_image_front = cv2.resize(cv_image_front, (self.orig_width, self.orig_height))
            # cv_image_hdr = cv2.resize(cv_image_hdr, (self.orig_width, self.orig_height))

            # Batch images
            # cv_image_batched = np.stack([cv_image_rear, cv_image_front, cv_image_hdr], axis=0)        
            cv_image_batched = np.stack([cv_image_rear, cv_image_front], axis=0)        
            # Perform segmentation model
            segmented_image, class_probabilities_batch = self.segment_image(cv_image_batched)

            with Timer('CPU Publishing'):

                # Hot fix to keep class_probs with 3 channels until multi-channel segmentation is implemented
                null_channel = np.zeros_like(class_probabilities_batch[:,:,:,:1])
                class_probabilities_batch = np.concatenate((class_probabilities_batch,
                                                            null_channel), axis=-1)
                                                            
                # Viz for the rear segmented image
                segmented_image = (class_probabilities_batch[0] * 255).astype(np.uint8)
                # hack to convert blue - green colours to red - yellow (in BGR)
                red_yellow = np.zeros_like(segmented_image)
                red_yellow[:,:,2] = segmented_image[:,:,0] + segmented_image[:,:,1]
                red_yellow[:,:,1] = segmented_image[:,:,1]
                segmented_msg = self.bridge.cv2_to_compressed_imgmsg(red_yellow)
                #set the frame id
                segmented_msg.header.frame_id = msg_rear.header.frame_id
                segmented_msg.header.stamp = msg_rear.header.stamp

                # Convert class_probabilities to img messages                    
                class_prob_msg_rear = self.bridge.cv2_to_imgmsg(class_probabilities_batch[0], encoding='passthrough')
                class_prob_msg_front = self.bridge.cv2_to_imgmsg(class_probabilities_batch[1], encoding='passthrough')
                # class_prob_msg_hdr = self.bridge.cv2_to_imgmsg(class_probabilities_batch[2], encoding='passthrough')

                # Set the frame ID and timestamp
                class_prob_msg_rear.header.frame_id = msg_rear.header.frame_id
                class_prob_msg_rear.header.stamp = msg_rear.header.stamp

                class_prob_msg_front.header.frame_id = msg_front.header.frame_id
                class_prob_msg_front.header.stamp = msg_front.header.stamp

                # class_prob_msg_hdr.header.frame_id = msg_hdr.header.frame_id
                # class_prob_msg_hdr.header.stamp = msg_hdr.header.stamp
                
                # Publish the class probabilities
                self.class_probs_rear_pub.publish(class_prob_msg_rear)
                self.class_probs_front_pub.publish(class_prob_msg_front)
                # self.class_probs_hdr_pub.publish(class_prob_msg_hdr)

                self.segmented_image_pub.publish(segmented_msg)

                # Publish the camera info
                camera_info_msg_rear = self.camera_info_msg_rear
                camera_info_msg_rear.header.stamp = msg_rear.header.stamp

                camera_info_msg_front = self.camera_info_msg_front
                camera_info_msg_front.header.stamp = msg_front.header.stamp

                # camera_info_msg_hdr = self.camera_info_msg_hdr
                # camera_info_msg_hdr.header.stamp = msg_hdr.header.stamp

                self.segmented_info_rear_pub.publish(camera_info_msg_rear)
                self.segmented_info_front_pub.publish(camera_info_msg_front)
                # self.segmented_info_hdr_pub.publish(camera_info_msg_hdr)

    def segment_image(self, image):
        '''Perform segmentation on the batch of images.
        
        Args:
            image: np.array of shape (batch_size, height, width, 3)
            
            Returns:
            np.array of shape (batch_size, height, width, 3): 0-255 segmented image colors
            np.array of shape (batch_size, height, width, num_labels): 0-1 class probabilities'''

        # Preprocess image and run inference
        with Timer('Preprocessing time'):
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        with Timer('Inference time'):    
            outputs = self.model(**inputs)
    
        logits = outputs.logits # shape (batch_size, num_labels, height/4, width/4)
        
        # Removed for performance optimization
        # First, rescale logits to original image size

        # upsampled_logits = nn.functional.interpolate(
        #     logits,
        #     size=image.size[::-1], # (height, width)
        #     mode='bilinear',
        #     align_corners=False
        # )

        # Convert to probabilities for each class
        upsampled_probs = nn.functional.softmax(logits, dim=1)

        # Permute 
        upsampled_probs = upsampled_probs.permute(0, 2, 3, 1)

        # Second, apply argmax on the class dimension
        pred_seg = logits.argmax(dim=1)[0]
        #ovelay the segmentation on the original image
        
        seg_overlay = self.get_seg_overlay(image, pred_seg)

        return seg_overlay.cpu().numpy(), upsampled_probs.cpu().numpy()

    
    def get_seg_overlay(self, image, seg):
        color_seg = torch.zeros((seg.shape[0], seg.shape[1], 3), dtype=torch.uint8, device=self.device) # height, width, 3
        color_seg = self.palette[seg.reshape(-1)]
        color_seg = color_seg.reshape(seg.shape[0], seg.shape[1], 3)
        return color_seg

if __name__ == '__main__':
    try:
        segmentation_node = SegmentationNode()
        print("Segmentation node started")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

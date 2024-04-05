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

        self.palette = torch.tensor([
                            [255, 0, 0],      # traversable
                            [255, 255, 0],      # untraversable
                            [0, 0, 0],          # unlabeled
                            ], 
                            dtype=torch.uint8,
                            device=self.device)

        # Publisher
        self.segmented_image_pub = rospy.Publisher('/traversable_segmentation/segmentation/compressed', CompressedImage, queue_size=1)
        self.class_probabilities_pub = rospy.Publisher('/traversable_segmentation/class_probabilities', Image, queue_size=1)
        self.segmented_info_pub = rospy.Publisher('/traversable_segmentation/camera_info', CameraInfo, queue_size=1)
        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Initialize model here
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH, device_map = self.device)
        self.model.eval()


        # # Subscribers
        self.image_sub = rospy.Subscriber('/wide_angle_camera_rear/image_color_rect/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        
        self.camera_info_msg = rospy.wait_for_message('/wide_angle_camera_rear/camera_info', CameraInfo)
        self.camera_info_msg.K = [k/4 for k in self.camera_info_msg.K]
        self.camera_info_msg.K[-1] = 1
        self.camera_info_msg.P = [p/4 for p in self.camera_info_msg.P]
        self.camera_info_msg.P[-2] = 1
        self.camera_info_msg.height = self.camera_info_msg.height/4
        self.camera_info_msg.width = self.camera_info_msg.width/4 




        # self.info_sub = rospy.Subscriber(, self.info_callback, queue_size=1)
        #Subscribers hdr camera //v4l2_camera/image_raw_throttle/compressed
        # self.image_sub = rospy.Subscriber('/v4l2_camera/image_raw_throttle/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        # self.info_sub = rospy.Subscriber('/v4l2_camera/camera_info_throttle', CameraInfo, self.info_callback, queue_size=1)
        # fixed topic /hdr_camera/image_raw/compressed
        # self.image_sub = rospy.Subscriber('/hdr_camera/image_raw/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        # self.info_sub = rospy.Subscriber('/hdr_camera/camera_info', CameraInfo, self.info_callback, queue_size=1)

    @torch.no_grad()
    def image_callback(self, msg):
        with Timer('Segmentation Callback'):

            #fix the frame id if it is the hdr camera
            if msg.header.frame_id == '':
                msg.header.frame_id = 'hdr_cam'          
            
            # Convert compressed image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Color conversion
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # Perform segmentation model
            segmented_image, class_probabilities = self.segment_image(cv_image)

            with Timer('CPU Publishing'):

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

                camera_info_msg = self.camera_info_msg
                camera_info_msg.header.stamp = msg.header.stamp
                self.segmented_info_pub.publish(camera_info_msg)


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
    

    def segment_image(self, image):
        # Perform segmentation 
        # conver to PIL image on device
        image = PIL.Image.fromarray(image)

        # Preprocess image
        with Timer('Preprocessing time'):
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        with Timer('Inference time'):    
            outputs = self.model(**inputs)
    
        logits = outputs.logits # shape (batch_size, num_labels, height/4, width/4)
        print(logits.shape, inputs["pixel_values"].shape, image.size[::-1])
        # First, rescale logits to original image size

        # upsampled_logits = nn.functional.interpolate(
        #     logits,
        #     size=image.size[::-1], # (height, width)
        #     mode='bilinear',
        #     align_corners=False
        # )

        # Convert to probabilities for each class
        upsampled_probs = nn.functional.softmax(logits, dim=1)
        # as a numpy array
        upsampled_probs = upsampled_probs[0].permute(1, 2, 0)

        # Second, apply argmax on the class dimension
        pred_seg = logits.argmax(dim=1)[0]
        #ovelay the segmentation on the original image
        
        seg_overlay = self.get_seg_overlay(image, pred_seg)

        return seg_overlay.cpu().numpy(), upsampled_probs.cpu().numpy()
        # #logit array return
        # upsampled_logits = upsampled_logits[0].detach().numpy().transpose(1, 2, 0)
        # return upsampled_logits
    
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

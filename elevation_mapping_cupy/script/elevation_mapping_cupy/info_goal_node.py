#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
import rospy
from geometry_msgs.msg import PoseStamped
from grid_map_msgs.msg import GridMap
import math



class InfoGoalNode:
    def __init__(self):
        print('Starting info_goal_node')
        rospy.init_node('info_goal_node')

        self.map_sub = rospy.Subscriber('/elevation_mapping/traversable_segmentation', GridMap, self.map_callback)
        self.goal_pub = rospy.Publisher('/info_goal_pose', PoseStamped, queue_size=10)

    def map_callback(self, msg):
        # Extract the info_goal layer from the grid map
        info_goal_ind= msg.layers.index('info_goal')
        info_goal_layer : tuple = msg.data[info_goal_ind].data
        col_dim = msg.data[info_goal_ind].layout.dim[0].size #TODO:consistent?
        row_dim = msg.data[info_goal_ind].layout.dim[1].size

        # Find the position of the one hot encoded cell
        goal_index = info_goal_layer.index(1)

        # Grid map center in header-defined frame
        center_x = msg.info.pose.position.x
        center_y = msg.info.pose.position.y

        resolution = msg.info.resolution
        length_x = msg.info.length_x
        length_y = msg.info.length_y

        # Calculate the position of the goal cell
        goal_x = center_x + resolution * (goal_index % col_dim)
        goal_y = center_y + resolution * (goal_index // col_dim)
        

        # Calculate the orientation of the goal pose
        yaw = 0
        quaternion = quaternion_from_euler(0, 0, yaw)


        # Create a PoseStamped message
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'odom'
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.orientation.x = quaternion[0]
        goal_pose.pose.orientation.y = quaternion[1]
        goal_pose.pose.orientation.z = quaternion[2]
        goal_pose.pose.orientation.w = quaternion[3]

        # Publish the goal pose
        self.goal_pub.publish(goal_pose)

if __name__ == '__main__':
    try:
        
        node = InfoGoalNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
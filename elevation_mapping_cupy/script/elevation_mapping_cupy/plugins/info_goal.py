from typing import List
import cupy as cp
from cupyx.scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import distance_transform_edt
from elevation_mapping_cupy.plugins.plugin_manager import PluginBase
import rospy
import geometry_msgs.msg as geo_msg
import cv2
import time

class FrontierExplorer(PluginBase):
    def __init__(self, 
                cell_n: int = 100,
                robot_size: int = 1, 
                 robot_cell_sight: int = 100
                 ):
        """This is a filter to find the optimal frontier cell to explore.'''

        Args:
            robot_size (int): The size of the robot in grid cells.
            robot_cell_sight (int): The sight distance of the robot in grid cells in one direction.
        """
        super().__init__()
        self.one_hot_frontier = cp.zeros((cell_n, cell_n), dtype=cp.float32)
        self.robot_size = robot_size
        self.robot_cell_sight = robot_cell_sight

        # TODO: move to params
        self.occupancy_layer_name = "traversable_fusion_min_layer"
        self.occupied_threshhold = 0.3

        #testing ros publisher
        self.pub = rospy.Publisher('frontier_goal', geo_msg.PoseStamped, queue_size=10)



    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """
        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            *args ():
        Returns:
            cp.ndarray:
        """
        print()
        print('Starting timer for frontier exploration.')
        self.start = time.perf_counter()
        # Get the elevation map elevation layer and set it as the grid map
        # Perform the frontier exploration and return the optimal frontier cell
        valid_layer_ind = layer_names.index('is_valid')
        valid_layer = elevation_map[valid_layer_ind]
        #erosion to remove small areas of 0s
        eroded_valid_layer = binary_erosion(valid_layer, iterations=5, brute_force=True)
        eroded_free_cells = cp.where(eroded_valid_layer, 0, cp.nan)
        eroded_bit_true = cp.where(eroded_valid_layer, 1, 0)
        
        # TODO: Occupancy grid 1s filled with trav layer selected in params
        occupancy_layer_ind = plugin_layer_names.index(self.occupancy_layer_name)
        occupied_cells = plugin_layers[occupancy_layer_ind]
        occupied_cells = cp.where(occupied_cells < self.occupied_threshhold, 1, 0)

        self.grid_map = eroded_free_cells + occupied_cells

        optimal_frontier = self.find_optimal_frontier()
        self.one_hot_frontier = cp.zeros_like(self.one_hot_frontier)
        if optimal_frontier is not None:
            window = (max(0,optimal_frontier[0] -5), 
                        min(optimal_frontier[0] + 5, self.one_hot_frontier.shape[0]),
                        max(0,optimal_frontier[1] -5),
                        min(optimal_frontier[1] + 5, self.one_hot_frontier.shape[1]))
            self.one_hot_frontier[window[0]:window[1], window[2]:window[3]] = 1

            self.pub_pose_from_frontier(optimal_frontier)
            disp_grid = cp.where(self.grid_map == 0, 0.5,self.grid_map)*255
            cv2.imwrite('frontier.png', cp.asnumpy(cp.clip(self.one_hot_frontier*255 + disp_grid, 0, 255)).astype('uint8'))
            
        elapsed = time.perf_counter() - self.start
        print(f"Total exploration took {elapsed:0.3f} seconds.")
        return cp.array(self.one_hot_frontier)

    def find_frontiers(self):
        # Find frontier cells (unknown cells that are adjacent to free cells, not occupied cells)
        unknown_cells = cp.isnan(self.grid_map)
        # Dilate occupied cells to account for the robot size
        occupied_cells = self.grid_map == 1
        occupied_cells_dilated = binary_dilation(occupied_cells, iterations=self.robot_size)

        free_cells = self.grid_map == 0
        free_cells_dilated = binary_dilation(free_cells, iterations=1)
        frontier_dist = distance_transform_edt(unknown_cells.get(), return_distances=True, return_indices=False)
        frontier_cells = (cp.array(frontier_dist) <= 1) & unknown_cells & free_cells_dilated & ~occupied_cells_dilated
        return cp.argwhere(frontier_cells)
        
    def calculate_information_gain(self, frontier):
        # Calculate number of unknown cells within robot_cell_sight of the frontier cell. 
        # Includes the unknowns cells outside the grid_map too. Simulate with padding.

        explore_grid_map = cp.pad(self.grid_map, self.robot_cell_sight, mode='constant', constant_values=cp.nan)
        unknown_cells = cp.isnan(explore_grid_map)
        unknown_cells = unknown_cells[frontier[0]:frontier[0] + 2 * self.robot_cell_sight + 1,
                                        frontier[1]:frontier[1] + 2 * self.robot_cell_sight + 1]
        
        return cp.sum(unknown_cells)


    def find_optimal_frontier(self):
        start_fontiers = time.perf_counter()
        frontiers = self.find_frontiers()
        elapsed_frontiers = time.perf_counter() - start_fontiers
        print(f"Frontier cells found in {elapsed_frontiers:0.3f} seconds.")
        # print("Frontiers:", frontiers)
        if len(frontiers) == 0:
            return None
        start_info_gain = time.perf_counter()
        max_information_gain = -1
        optimal_frontier = None
        for frontier in frontiers:
            information_gain = self.calculate_information_gain(frontier)
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                optimal_frontier = frontier
        print("Frontier:", optimal_frontier, "Information gain:", max_information_gain)
        elapsed_info_gain = time.perf_counter() - start_info_gain
        print(f"Optimal frontier found in {elapsed_info_gain:0.3f} seconds.")
        return optimal_frontier
    
    def pub_pose_from_frontier(self, frontier):
        #publish the goal pose from the optimal frontier
        # pass
        goal_pose = geo_msg.PoseStamped()
        goal_pose.header.frame_id = "base_inverted"
        goal_pose.pose.position.x = (-(frontier[1]-100))*8/200
        goal_pose.pose.position.y = (frontier[0]-100)*8/200
        goal_pose.pose.position.z = 0
        goal_pose.pose.orientation.x = 0
        goal_pose.pose.orientation.y = 0
        goal_pose.pose.orientation.z = 0
        goal_pose.pose.orientation.w = 1
        self.pub.publish(goal_pose)

# if __name__ == "__main__":

#     # Example usage:
#     grid_map = cp.array([[cp.nan,   0,      0,          cp.nan],
#                         [cp.nan,   1,      cp.nan,     cp.nan],
#                         [0,        0,      0,          1],
#                         [0,        cp.nan, cp.nan,     1]])

#     robot_size = 1
#     robot_cell_sight = 2  # Assuming the robot can see 2 grid cells in each direction, not exactly a radius but consistent with gridmap

#     explorer = FrontierExplorer(grid_map, robot_size, robot_cell_sight)
#     optimal_frontier = explorer.find_optimal_frontier()
#     print("Optimal frontier:", optimal_frontier)

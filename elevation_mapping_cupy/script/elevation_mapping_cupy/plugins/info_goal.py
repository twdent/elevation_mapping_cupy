from typing import List
import cupy as cp
from cupyx.scipy.ndimage import binary_dilation, binary_erosion, convolve
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import convolve as scipy_convolve
from elevation_mapping_cupy.plugins.plugin_manager import PluginBase
# import rospy
# import geometry_msgs.msg as geo_msg
import cv2
import time
import torch

class FrontierExplorer(PluginBase):
    def __init__(self, 
                cell_n: int = 100,
                robot_size: int = 1, 
                robot_cell_sight: int = 100,
                occupancy_layer_name: str = 'traversability',
                occupied_threshold: float = 0.3,
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
        self.occupancy_layer_name = str(occupancy_layer_name)
        self.occupied_threshold = float(occupied_threshold)

        self.information_gain_arr = cp.zeros((cell_n, cell_n), dtype=cp.float32)
        self.sdf_frontier = cp.zeros((cell_n, cell_n), dtype=cp.float32)

        #testing ros publisher
        # self.pub = rospy.Publisher('frontier_goal', geo_msg.PoseStamped, queue_size=10)

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_layers: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            semantic_layers (cupy._core.core.ndarray):
            semantic_layer_names (List[str]):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        # print()
        # print('Starting timer for frontier exploration.')
        
        # Get the elevation map elevation layer and set it as the grid map
        # Perform the frontier exploration and return the optimal frontier cell
        valid_layer = self.get_layer_data(elevation_map, layer_names, plugin_layers, 
                                          plugin_layer_names, semantic_layers, semantic_layer_names, 
                                          'is_valid')
        #erosion to remove small areas of 0s
        eroded_valid_layer = binary_erosion(valid_layer, iterations=1, brute_force=True)
        eroded_free_cells = cp.where(eroded_valid_layer, 0, cp.nan)
        eroded_bit_true = cp.where(eroded_valid_layer, 1, 0)
        
        # TODO: Occupancy grid 1s filled with trav layer selected in params
        occupancy_layer = self.get_layer_data(elevation_map, layer_names, plugin_layers,
                                                plugin_layer_names, semantic_layers, semantic_layer_names,
                                                self.occupancy_layer_name)
        occupied_cells = cp.where(occupancy_layer < self.occupied_threshold, 1, 0)

        self.grid_map = eroded_free_cells + occupied_cells

        # TODO: SDF layer selected in params
        sdf_layer_ind = plugin_layer_names.index('sdf_layer0.1')
        sdf_layer = plugin_layers[sdf_layer_ind]

        optimal_frontier = self.find_optimal_frontier()
        optimal_frontier = self.find_optimal_frontier_sdf(sdf_layer)

        self.one_hot_frontier = cp.zeros_like(self.one_hot_frontier)
        #for vizualization
        if optimal_frontier is not None:
            k=1
            window = (max(0,optimal_frontier[0] -k), 
                        min(optimal_frontier[0] + k, self.one_hot_frontier.shape[0]),
                        max(0,optimal_frontier[1] -k),
                        min(optimal_frontier[1] + k, self.one_hot_frontier.shape[1]))
            self.one_hot_frontier[window[0]:window[1], window[2]:window[3]] = 1

            self.pub_pose_from_frontier(optimal_frontier)
            
            r_chan = cp.asnumpy(cp.where(self.grid_map == 1, 255,0)).astype('uint8')
            g_chan = cp.asnumpy(self.one_hot_frontier*255).astype('uint8')
            b_chan = cp.asnumpy(cp.where(self.grid_map == 0, 0,255)).astype('uint8')
            disp_img = cv2.merge((r_chan, g_chan, b_chan))
            cv2.imwrite('frontier.png', disp_img)
        else:
            print("No frontier found.")    

        return cp.array(self.one_hot_frontier)
        # return cp.array(self.sdf_frontier)
        return cp.array(self.information_gain_arr)

    def find_frontiers(self,circle_mask=True):
        # Find frontier cells (unknown cells that are adjacent to free cells, not occupied cells)
        unknown_cells = cp.isnan(self.grid_map)
        # Dilate occupied cells to account for the robot size
        occupied_cells = self.grid_map == 1
        occupied_cells_dilated = binary_dilation(occupied_cells, iterations=self.robot_size)

        free_cells = self.grid_map == 0
        free_cells_dilated = binary_dilation(free_cells, iterations=1)
        frontier_dist = distance_transform_edt(unknown_cells.get(), return_distances=True, return_indices=False)
        frontier_cells = (cp.array(frontier_dist) <= 1) & unknown_cells & free_cells_dilated & ~occupied_cells_dilated

        if circle_mask:
            # mask to disregard frontier cells that are too close to the corners of the grid
            grid_size = self.grid_map.shape[0]
            y, x = cp.ogrid[-grid_size//2:grid_size//2, -grid_size//2:grid_size//2]
            mask = x**2 + y**2 <= (grid_size//2)**2
            frontier_cells = frontier_cells & mask

        return frontier_cells
        
    def calculate_information_gain(self, frontier,padded_unknown_cells):
        # Calculate number of unknown cells within robot_cell_sight of the frontier cell. 
        # Includes the unknowns cells outside the grid_map too. Simulate with padding.

        unknown_cells = padded_unknown_cells[frontier[0]:frontier[0] + 2 * self.robot_cell_sight + 1,
                                        frontier[1]:frontier[1] + 2 * self.robot_cell_sight + 1]
        
        return cp.sum(unknown_cells)


    def find_optimal_frontier(self):
        '''Find the optimal frontier cell to explore based on the information gain.'''
 

        frontier_mask = self.find_frontiers(circle_mask=True)
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # torch.cuda.synchronize()
        # start_event.record()
        # end_event.record()
        # torch.cuda.synchronize()
        # elapsed_time = start_event.elapsed_time(end_event)
        # print(f"find_frontiers took {elapsed_time:0.3f} ms.")

        if len(cp.argwhere(frontier_mask)) == 0:
            return None
        max_information_gain = -1
        optimal_frontier = None
        
        # kernel approximating a circle of radius robot_cell_sight
        kernel = cp.zeros((2 * self.robot_cell_sight + 1, 2 * self.robot_cell_sight + 1))
        for i in range(2 * self.robot_cell_sight + 1):
            for j in range(2 * self.robot_cell_sight + 1):
                if (i - self.robot_cell_sight) ** 2 + (j - self.robot_cell_sight) ** 2 <= self.robot_cell_sight ** 2:
                    kernel[i, j] = 1

        info_cells = cp.where(cp.isnan(self.grid_map), 1, 0)
        self.information_gain_arr = convolve(info_cells, kernel, mode='constant', cval=1)

        # Mask to only consider frontier cells
        valid_info_gain = self.information_gain_arr * frontier_mask
        optimal_frontier = cp.unravel_index(cp.argmax(valid_info_gain), valid_info_gain.shape)
        max_information_gain = cp.max(valid_info_gain)
        
        # explore_grid_map = cp.pad(self.grid_map, self.robot_cell_sight, mode='constant', constant_values=cp.nan)
        # padded_unknown_cells = cp.isnan(explore_grid_map)

        # for frontier in frontiers: #TODO:slow to test all, could reduce somehow
        #     information_gain = cp.sum(padded_unknown_cells[frontier[0]:frontier[0] + 2 * self.robot_cell_sight + 1,
        #                                 frontier[1]:frontier[1] + 2 * self.robot_cell_sight + 1])
        #     if information_gain > max_information_gain:
        #         max_information_gain = information_gain
        #         optimal_frontier = frontier

        # elapsed_info_gain = time.perf_counter() - start_info_gain
        # print("Frontier:", optimal_frontier, "Information gain:", max_information_gain)
        # print(f"Optimal loop free frontier found in {elapsed_info_gain:0.3f} seconds.")
        return optimal_frontier
    
    def find_optimal_frontier_sdf(self,sdf_layer):
        '''Finds optimal frontier cell to explore, with information gain weighted by the SDF layer.'''
        #TODO: Correct timing of this funciton
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # torch.cuda.synchronize()
        # start_event.record()
        frontier_mask = self.find_frontiers(circle_mask=True)
        # end_event.record()
        # torch.cuda.synchronize()
        # elapsed_time = start_event.elapsed_time(end_event)
        # print(f"Frontier cells found in {elapsed_time:0.3f} ms.")

        if len(cp.argwhere(frontier_mask)) == 0:
            return None
        max_information_gain = -1
        optimal_frontier = None
        
        # kernel approximating a circle of radius robot_cell_sight
        kernel = cp.zeros((2 * self.robot_cell_sight + 1, 2 * self.robot_cell_sight + 1))
        for i in range(2 * self.robot_cell_sight + 1):
            for j in range(2 * self.robot_cell_sight + 1):
                if (i - self.robot_cell_sight) ** 2 + (j - self.robot_cell_sight) ** 2 <= self.robot_cell_sight ** 2:
                    kernel[i, j] = 1
        
        info_cells = cp.where(cp.isnan(self.grid_map), 1, 0)
        

        self.information_gain_arr = convolve(info_cells, kernel, mode='constant', cval=1)


        norm_sdf = (sdf_layer - cp.min(sdf_layer))/(cp.max(sdf_layer) - cp.min(sdf_layer))
        # self.information_gain_arr = (self.information_gain_arr - cp.min(self.information_gain_arr))/(cp.max(self.information_gain_arr) - cp.min(self.information_gain_arr))
        # Mask to only consider frontier cells
        valid_info_gain = self.information_gain_arr * frontier_mask
        
        self.sdf_frontier = valid_info_gain * norm_sdf
        # self.sdf_frontier = valid_info_gain + norm_sdf
        optimal_frontier = cp.unravel_index(cp.argmax(self.sdf_frontier), self.sdf_frontier.shape)
        max_information_gain = cp.max(self.sdf_frontier)

        print("Frontier:", optimal_frontier, "Information gain:", max_information_gain)
        return optimal_frontier


    def pub_pose_from_frontier(self, frontier):
        #publish the goal pose from the optimal frontier
        pass
        # goal_pose = geo_msg.PoseStamped()
        # goal_pose.header.frame_id = "odom"
        # goal_pose.pose.position.x = (-(frontier[1]-100))*8/200
        # goal_pose.pose.position.y = (frontier[0]-100)*8/200
        # goal_pose.pose.position.z = 0
        # goal_pose.pose.orientation.x = 0
        # goal_pose.pose.orientation.y = 0
        # goal_pose.pose.orientation.z = 0
        # goal_pose.pose.orientation.w = 1
        # self.pub.publish(goal_pose)

if __name__ == "__main__":

    # Example usage:
    grid_map = cp.array([[cp.nan,   0,      0,          cp.nan],
                        [cp.nan,   1,      cp.nan,     cp.nan],
                        [0,        0,      0,          1],
                        [0,        cp.nan, cp.nan,     1]])

    robot_size = 1
    robot_cell_sight = 2  # Assuming the robot can see 2 grid cells in each direction, not exactly a radius but consistent with gridmap

    explorer = FrontierExplorer()
    explorer.grid_map = grid_map

  
    optimal_frontier = explorer.find_optimal_frontier()
    print("Optimal frontier:", optimal_frontier)

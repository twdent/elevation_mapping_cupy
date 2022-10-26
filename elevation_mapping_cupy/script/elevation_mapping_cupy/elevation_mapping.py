#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import os
from typing import List, Any, Tuple, Union

import numpy as np
import threading
import subprocess

from .traversability_filter import get_filter_chainer, get_filter_torch
from .parameter import Parameter
from .custom_kernels import add_points_kernel, add_color_kernel, color_average_kernel
from .custom_kernels import sum_kernel
from .custom_kernels import error_counting_kernel
from .custom_kernels import average_map_kernel
from .custom_kernels import dilation_filter_kernel
from .custom_kernels import normal_filter_kernel
from .custom_kernels import polygon_mask_kernel
from .map_initializer import MapInitializer
from .plugins.plugin_manager import PluginManager
from .semantic_map import SemanticMap

from .traversability_polygon import (
    get_masked_traversability,
    is_traversable,
    calculate_area,
    transform_to_map_position,
    transform_to_map_index,
)

import cupy as cp

xp = cp
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)


class ElevationMap:
    """
    Core elevation mapping class.
    """

    def __init__(self, param: Parameter):
        """

        Args:
            param (elevation_mapping_cupy.parameter.Parameter):
        """
        self.param = param

        self.resolution = param.resolution
        self.center = xp.array([0, 0, 0], dtype=float)
        self.map_length = param.map_length
        # +2 is a border for outside map
        # TODO
        self.cell_n = int(round(self.map_length / self.resolution)) + 2
        self.param.cell_n = self.cell_n

        self.map_lock = threading.Lock()
        self.additional_layers = dict(zip(self.param.additional_layers, self.param.fusion_algorithms))
        # self.amount_additional_layers = len(self.param.additional_layers)
        self.semantic_map = SemanticMap(self.param, self.additional_layers)
        # layers: elevation, variance, is_valid, traversability, time, upper_bound, is_upper_bound
        self.elevation_map = xp.zeros((7, self.cell_n, self.cell_n))
        self.layer_names = [
            "elevation",
            "variance",
            "is_valid",
            "traversability",
            "time",
            "upper_bound",
            "is_upper_bound",
        ]
        # self.layer_names.extend(list(self.additional_layers.keys()))

        # self.unique_fusion = []
        # for x in list(self.additional_layers.values()):
        #     if x not in self.unique_fusion:
        #         self.unique_fusion.append(x)
        # print("Fusion algorithms running: ",self.unique_fusion)
        # buffers
        self.traversability_buffer = xp.full((self.cell_n, self.cell_n), xp.nan)
        self.normal_map = xp.zeros((3, self.cell_n, self.cell_n))
        # Initial variance
        self.initial_variance = param.initial_variance
        self.elevation_map[1] += self.initial_variance
        self.elevation_map[3] += 1.0

        # overlap clearance
        cell_range = int(self.param.overlap_clear_range_xy / self.resolution)
        cell_range = np.clip(cell_range, 0, self.cell_n)
        self.cell_min = self.cell_n // 2 - cell_range // 2
        self.cell_max = self.cell_n // 2 + cell_range // 2

        # Initial mean_error
        self.mean_error = 0.0
        self.additive_mean_error = 0.0

        self.compile_kernels()
        # self.compile_additional_kernels()
        self.semantic_map.compile_kernels()

        weight_file = subprocess.getoutput('echo "' + param.weight_file + '"')
        param.load_weights(weight_file)

        if param.use_chainer:
            self.traversability_filter = get_filter_chainer(param.w1, param.w2, param.w3, param.w_out)
        else:
            self.traversability_filter = get_filter_torch(param.w1, param.w2, param.w3, param.w_out)
        self.untraversable_polygon = xp.zeros((1, 2))

        # Plugins
        self.plugin_manager = PluginManager(cell_n=self.cell_n)
        plugin_config_file = subprocess.getoutput('echo "' + param.plugin_config_file + '"')
        self.plugin_manager.load_plugin_settings(plugin_config_file)

        self.map_initializer = MapInitializer(self.initial_variance, param.initialized_variance, xp=cp, method="points")
        # self.color_map = None
    #
    # def compile_additional_kernels(self) -> None:
    #     """
    #     Returns:
    #         None:
    #
    #     """
    #     if ("average"  in self.unique_fusion):
    #         print("Initialize fusion kernel")
    #         self.sum_kernel = sum_kernel(
    #             self.resolution,
    #             self.cell_n,
    #             self.cell_n,
    #             self.param.sensor_noise_factor,
    #             self.param.min_valid_distance,
    #             self.param.max_height_range,
    #             self.param.ramped_height_range_a,
    #             self.param.ramped_height_range_b,
    #             self.param.ramped_height_range_c,
    #
    #         )
    #
    #     if ("color" in self.unique_fusion):
    #         print("Initialize color kernel")
    #         self.add_color_kernel = add_color_kernel(
    #             self.cell_n,
    #             self.cell_n,
    #         )
    #         self.color_average_kernel = color_average_kernel(self.cell_n,self.cell_n)

    # def get_fusion_of_pcl(self, channels: List[str]) -> List[str]:
    #     """Get all fusion algorithms that need to be applied to a specific pointcloud
    #
    #     Args:
    #         channels (List[str]):
    #     """
    #     fusion_list = []
    #     for channel in channels:
    #         x = self.additional_layers[channel]
    #         if x not in fusion_list:
    #             fusion_list.append(x)
    #     return fusion_list
    #
    # def get_indices_fusion(self, pcl_channels: List[str], fusion_alg: str):
    #     """Computes the indices that are used for the additional kernel update of the pcl and the elmap.
    #
    #     Args:
    #         pcl_channels (List[str]):
    #         fusion_alg (str):
    #
    #     Returns:
    #         Union[Tuple[List[int], List[int]], Tuple[cupy._core.core.ndarray, cupy._core.core.ndarray]]:
    #
    #
    #     """
    #     # this contains exactly the fusion alg type for each channel of the pcl
    #     pcl_val_list = [self.additional_layers[x] for x in pcl_channels]
    #     # this contains the indeces of the pointcloud where we have to perform a certain fusion
    #     pcl_indices = cp.array([idp+3 for idp, x in enumerate(pcl_val_list) if x == fusion_alg],dtype=np.int32)
    #     # create a list of indeces of the layers that will be updated by the pointcloud with specific fusion alg
    #     layer_indices = cp.array([], dtype=np.int32)
    #     for it, (key, val) in enumerate(self.additional_layers.items()):
    #         if key in pcl_channels and val == fusion_alg:
    #             layer_indices = cp.append(layer_indices,7+it ).astype(np.int32)
    #     return pcl_indices, layer_indices

    def clear(self):
        with self.map_lock:
            self.elevation_map *= 0.0
            # Initial variance
            self.elevation_map[1] += self.initial_variance
            self.semantic_map.map *= 0.0
        self.mean_error = 0.0
        self.additive_mean_error = 0.0

    def get_position(self, position):
        position[0][:] = xp.asnumpy(self.center)

    def move(self, delta_position):
        # Shift map using delta position.
        delta_position = xp.asarray(delta_position)
        delta_pixel = xp.round(delta_position[:2] / self.resolution)
        delta_position_xy = delta_pixel * self.resolution
        self.center[:2] += xp.asarray(delta_position_xy)
        self.center[2] += xp.asarray(delta_position[2])
        self.shift_map_xy(delta_pixel)
        self.shift_map_z(-delta_position[2])

    def move_to(self, position, R):
        # Shift map to the center of robot.
        self.base_rotation = xp.asarray(R)
        position = xp.asarray(position)
        delta = position - self.center
        delta_pixel = xp.around(delta[:2] / self.resolution)
        delta_xy = delta_pixel * self.resolution
        self.center[:2] += delta_xy
        self.center[2] += delta[2]
        self.shift_map_xy(-delta_pixel)
        self.shift_map_z(-delta[2])

    def pad_value(self, x, shift_value, idx=None, value=0.0):
        """Create a padding of the map along x,y axis according to amount that has shifted.

        Args:
            x (cupy._core.core.ndarray):
            shift_value (cupy._core.core.ndarray):
            idx (Union[None, int, None, None]):
            value (float):
        """
        if idx is None:
            if shift_value[0] > 0:
                x[:, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[:, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[:, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[:, :, shift_value[1] :] = value
        else:
            if shift_value[0] > 0:
                x[idx, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[idx, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[idx, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[idx, :, shift_value[1] :] = value

    def shift_map_xy(self, delta_pixel):
        shift_value = delta_pixel.astype(cp.int)
        if cp.abs(shift_value).sum() == 0:
            return
        with self.map_lock:
            self.elevation_map = cp.roll(self.elevation_map, shift_value, axis=(1, 2))
            self.pad_value(self.elevation_map, shift_value, value=0.0)
            self.pad_value(self.elevation_map, shift_value, idx=1, value=self.initial_variance)

            self.semantic_map.map = cp.roll(self.semantic_map.map, shift_value, axis=(1, 2))
            self.pad_value(self.semantic_map.map, shift_value, value=0.0)

    def shift_map_z(self, delta_z):
        with self.map_lock:
            # elevation
            self.elevation_map[0] += delta_z
            # upper bound
            self.elevation_map[5] += delta_z

    def compile_kernels(self):
        # Compile custom cuda kernels.
        self.new_map = cp.zeros((self.elevation_map.shape[0], self.cell_n, self.cell_n))
        self.traversability_input = cp.zeros((self.cell_n, self.cell_n))
        self.traversability_mask_dummy = cp.zeros((self.cell_n, self.cell_n))
        self.min_filtered = cp.zeros((self.cell_n, self.cell_n))
        self.min_filtered_mask = cp.zeros((self.cell_n, self.cell_n))
        self.mask = cp.zeros((self.cell_n, self.cell_n))
        self.add_points_kernel = add_points_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.sensor_noise_factor,
            self.param.mahalanobis_thresh,
            self.param.outlier_variance,
            self.param.wall_num_thresh,
            self.param.max_ray_length,
            self.param.cleanup_step,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.cleanup_cos_thresh,
            self.param.ramped_height_range_a,
            self.param.ramped_height_range_b,
            self.param.ramped_height_range_c,
            self.param.enable_edge_sharpen,
            self.param.enable_visibility_cleanup,
        )
        self.error_counting_kernel = error_counting_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.sensor_noise_factor,
            self.param.mahalanobis_thresh,
            self.param.drift_compensation_variance_inlier,
            self.param.traversability_inlier,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.ramped_height_range_a,
            self.param.ramped_height_range_b,
            self.param.ramped_height_range_c,
        )
        self.average_map_kernel = average_map_kernel(
            self.cell_n, self.cell_n, self.param.max_variance, self.initial_variance
        )

        self.dilation_filter_kernel = dilation_filter_kernel(self.cell_n, self.cell_n, self.param.dilation_size)
        self.dilation_filter_kernel_initializer = dilation_filter_kernel(
            self.cell_n, self.cell_n, self.param.dilation_size_initialize
        )
        self.polygon_mask_kernel = polygon_mask_kernel(self.cell_n, self.cell_n, self.resolution)
        self.normal_filter_kernel = normal_filter_kernel(self.cell_n, self.cell_n, self.resolution)

    def shift_translation_to_map_center(self, t):
        t -= self.center

    def update_map_with_kernel(self, points_all, channels, R, t, position_noise, orientation_noise):
        self.new_map *= 0.0
        error = cp.array([0.0], dtype=cp.float32)
        error_cnt = cp.array([0], dtype=cp.float32)
        points = points_all[:,:3]
        # additional_fusion = self.get_fusion_of_pcl(channels)
        with self.map_lock:
            self.shift_translation_to_map_center(t)
            self.error_counting_kernel(
                self.elevation_map,
                points,
                cp.array([0.0]),
                cp.array([0.0]),
                R,
                t,
                self.new_map,
                error,
                error_cnt,
                size=(points.shape[0]),
            )
            if (
                self.param.enable_drift_compensation
                and error_cnt > self.param.min_height_drift_cnt
                and (
                    position_noise > self.param.position_noise_thresh
                    or orientation_noise > self.param.orientation_noise_thresh
                )
            ):
                self.mean_error = error / error_cnt
                self.additive_mean_error += self.mean_error
                if np.abs(self.mean_error) < self.param.max_drift:
                    self.elevation_map[0] += self.mean_error * self.param.drift_compensation_alpha
            self.add_points_kernel(
                cp.array([0.0]),
                cp.array([0.0]),
                R,
                t,
                self.normal_map,
                points,
                self.elevation_map,
                self.new_map,
                size=(points.shape[0]),
            )
            self.average_map_kernel(self.new_map, self.elevation_map, size=(self.cell_n * self.cell_n))

            # self.update_additional_layers(additional_fusion, points_all, channels, R, t)
            self.semantic_map.update_layers(points_all, channels, R, t)

            if self.param.enable_overlap_clearance:
                self.clear_overlap_map(t)
            # dilation before traversability_filter
            self.traversability_input *= 0.0
            self.dilation_filter_kernel(
                self.elevation_map[5],
                self.elevation_map[2] + self.elevation_map[6],
                self.traversability_input,
                self.traversability_mask_dummy,
                size=(self.cell_n * self.cell_n),
            )
            # calculate traversability
            traversability = self.traversability_filter(self.traversability_input)
            self.elevation_map[3][3:-3, 3:-3] = traversability.reshape(
                (traversability.shape[2], traversability.shape[3])
            )

        # calculate normal vectors
        self.update_normal(self.traversability_input)

    # def update_additional_layers(self, additional_fusion, points_all,channels,R,t):
    #     if "average" in additional_fusion:
    #         pcl_ids, layer_ids = self.get_indices_fusion(channels, "average")
    #         self.sum_kernel(
    #             points_all,
    #             cp.array([0.0]),
    #             cp.array([0.0]),
    #             R,
    #             t,
    #             self.normal_map,
    #             pcl_ids,
    #             layer_ids,
    #             cp.array([points_all.shape[1],pcl_ids.shape[0]],dtype=np.int32),
    #             self.elevation_map,
    #             self.new_map,
    #             size=(points_all.shape[0]),
    #         )
    #     if "color" in additional_fusion:
    #         pcl_ids, layer_ids = self.get_indices_fusion(channels, "color")
    #         if self.color_map is None:
    #             self.color_map = cp.zeros((1+3*layer_ids.shape[0], self.cell_n, self.cell_n), dtype=np.uint32)
    #         points_all = cp.asarray(cp.float32(points_all.get()))
    #         self.add_color_kernel(
    #             points_all,
    #             cp.array([0.0]),
    #             cp.array([0.0]),
    #             R,
    #             t,
    #             self.normal_map,
    #             pcl_ids,
    #             layer_ids,
    #             cp.array([points_all.shape[1],pcl_ids.shape[0]],dtype=np.int32),
    #             self.elevation_map,
    #             self.color_map,
    #             size=(points_all.shape[0]),
    #         )
    #         self.color_average_kernel(self.color_map,
    #             pcl_ids,
    #             layer_ids,
    #             cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=np.int32),
    #             self.elevation_map,
    #             size=(self.cell_n * self.cell_n),
    #         )

    def clear_overlap_map(self, t):
        # Clear overlapping area around center
        height_min = t[2] - self.param.overlap_clear_range_z
        height_max = t[2] + self.param.overlap_clear_range_z
        near_map = self.elevation_map[:, self.cell_min : self.cell_max, self.cell_min : self.cell_max]
        valid_idx = ~cp.logical_or(near_map[0] < height_min, near_map[0] > height_max)
        near_map[0] = cp.where(valid_idx, near_map[0], 0.0)
        near_map[1] = cp.where(valid_idx, near_map[1], self.initial_variance)
        near_map[2] = cp.where(valid_idx, near_map[2], 0.0)
        valid_idx = ~cp.logical_or(near_map[5] < height_min, near_map[5] > height_max)
        near_map[5] = cp.where(valid_idx, near_map[5], 0.0)
        near_map[6] = cp.where(valid_idx, near_map[6], 0.0)
        self.elevation_map[:, self.cell_min : self.cell_max, self.cell_min : self.cell_max] = near_map

    def get_additive_mean_error(self):
        return self.additive_mean_error

    def update_variance(self):
        self.elevation_map[1] += self.param.time_variance * self.elevation_map[2]

    def update_time(self):
        self.elevation_map[4] += self.param.time_interval

    def update_upper_bound_with_valid_elevation(self):
        mask = self.elevation_map[2] > 0.5
        self.elevation_map[5] = cp.where(mask, self.elevation_map[0], self.elevation_map[5])
        self.elevation_map[6] = cp.where(mask, 0.0, self.elevation_map[6])

    def input(self, raw_points: cp._core.core.ndarray, channels: List[str], R: cp._core.core.ndarray,
              t: cp._core.core.ndarray,
              position_noise: int,
              orientation_noise: int) -> None:
        """Input the pointcloud and fuse the new measurements to update the elevation map.

        Args:
            raw_points (cupy._core.core.ndarray): 
            channels (List[str]): 
            R  (cupy._core.core.ndarray):
            t (cupy._core.core.ndarray): 
            position_noise (int): 
            orientation_noise (int): 

        Returns:
            None: 
        """
        # Update elevation map using point cloud input.
        raw_points = cp.asarray(raw_points)
        additional_channels = channels[3:]
        raw_points = raw_points[~cp.isnan(raw_points).any(axis=1)]
        self.update_map_with_kernel(raw_points, additional_channels, cp.asarray(R), cp.asarray(t), position_noise, orientation_noise)

    def update_normal(self, dilated_map):
        with self.map_lock:
            self.normal_map *= 0.0
            self.normal_filter_kernel(
                dilated_map, self.elevation_map[2], self.normal_map, size=(self.cell_n * self.cell_n)
            )

    def process_map_for_publish(self, input_map, fill_nan=False, add_z=False, xp=cp):
        m = input_map.copy()
        if fill_nan:
            m = xp.where(self.elevation_map[2] > 0.5, m, xp.nan)
        if add_z:
            m = m + self.center[2]
        return m[1:-1, 1:-1]

    def get_elevation(self):
        return self.process_map_for_publish(self.elevation_map[0], fill_nan=True, add_z=True)

    def get_variance(self):
        return self.process_map_for_publish(self.elevation_map[1], fill_nan=False, add_z=False)

    def get_traversability(self):
        traversability = cp.where(
            (self.elevation_map[2] + self.elevation_map[6]) > 0.5, self.elevation_map[3].copy(), cp.nan
        )
        self.traversability_buffer[3:-3, 3:-3] = traversability[3:-3, 3:-3]
        traversability = self.traversability_buffer[1:-1, 1:-1]
        return traversability

    def get_time(self):
        return self.process_map_for_publish(self.elevation_map[4], fill_nan=False, add_z=False)

    def get_upper_bound(self):
        if self.param.use_only_above_for_upper_bound:
            valid = cp.logical_or(
                cp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5), self.elevation_map[2] > 0.5
            )
        else:
            valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        upper_bound = cp.where(valid, self.elevation_map[5].copy(), cp.nan)
        upper_bound = upper_bound[1:-1, 1:-1] + self.center[2]
        return upper_bound

    def get_is_upper_bound(self):
        if self.param.use_only_above_for_upper_bound:
            valid = cp.logical_or(
                cp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5), self.elevation_map[2] > 0.5
            )
        else:
            valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        is_upper_bound = cp.where(valid, self.elevation_map[6].copy(), cp.nan)
        is_upper_bound = is_upper_bound[1:-1, 1:-1]
        return is_upper_bound

    def xp_of_array(self, array):
        if type(array) == cp.ndarray:
            return cp
        elif type(array) == np.ndarray:
            return np

    def copy_to_cpu(self, array, data, stream=None):
        if type(array) == np.ndarray:
            data[...] = array.astype(np.float32)
        elif type(array) == cp.ndarray:
            if stream is not None:
                data[...] = cp.asnumpy(array.astype(np.float32), stream=stream)
            else:
                data[...] = cp.asnumpy(array.astype(np.float32))

    def exists_layer(self, name):
        if name in self.layer_names:
            return True
        elif name in self.additional_layers:
            return True
        elif name in self.plugin_manager.layer_names:
            return True
        else:
            return False

    # def get_rgb(self):
    #     c = self.process_map_for_publish(self.elevation_map[7], fill_nan=False, add_z=False)
    #     c = xp.uint32(c.get())
    #     c.dtype = np.float32
    #     return c
    #
    # def get_semantic(self,name):
    #     idx = self.layer_names.index(name)
    #     c = self.process_map_for_publish(self.elevation_map[idx], fill_nan=False, add_z=False)
    #     return c

    def get_map_with_name_ref(self, name, data):
        use_stream = True
        xp = cp
        with self.map_lock:
            if name == "elevation":
                m = self.get_elevation()
                use_stream = False
            elif name == "variance":
                m = self.get_variance()
            elif name == "traversability":
                m = self.get_traversability()
            elif name == "time":
                m = self.get_time()
            elif name == "upper_bound":
                m = self.get_upper_bound()
            elif name == "is_upper_bound":
                m = self.get_is_upper_bound()
            elif name == "normal_x":
                m = self.normal_map.copy()[0, 1:-1, 1:-1]
            elif name == "normal_y":
                m = self.normal_map.copy()[1, 1:-1, 1:-1]
            elif name == "normal_z":
                m = self.normal_map.copy()[2, 1:-1, 1:-1]
            # elif name in self.additional_layers.keys():
            #     if self.additional_layers[name] == "color":
            #         m = self.get_rgb()
            #     else:
            #         m = self.get_semantic()

            elif name in self.additional_layers.keys():
                m = self.semantic_map.get_map_with_name(name)
            elif name in self.plugin_manager.layer_names:
                self.plugin_manager.update_with_name(name, self.elevation_map, self.layer_names)
                m = self.plugin_manager.get_map_with_name(name)
                p = self.plugin_manager.get_param_with_name(name)
                xp = self.xp_of_array(m)
                m = self.process_map_for_publish(m, fill_nan=p.fill_nan, add_z=p.is_height_layer, xp=xp)
            else:
                print("Layer {} is not in the map".format(name))
                return
        m = xp.flip(m, 0)
        m = xp.flip(m, 1)
        if use_stream:
            stream = cp.cuda.Stream(non_blocking=False)
        else:
            stream = None
        self.copy_to_cpu(m, data, stream=stream)

    def get_normal_maps(self):
        normal = self.normal_map.copy()
        normal_x = normal[0, 1:-1, 1:-1]
        normal_y = normal[1, 1:-1, 1:-1]
        normal_z = normal[2, 1:-1, 1:-1]
        maps = xp.stack([normal_x, normal_y, normal_z], axis=0)
        maps = xp.flip(maps, 1)
        maps = xp.flip(maps, 2)
        maps = xp.asnumpy(maps)
        return maps

    def get_normal_ref(self, normal_x_data, normal_y_data, normal_z_data):
        maps = self.get_normal_maps()
        self.stream = cp.cuda.Stream(non_blocking=True)
        normal_x_data[...] = xp.asnumpy(maps[0], stream=self.stream)
        normal_y_data[...] = xp.asnumpy(maps[1], stream=self.stream)
        normal_z_data[...] = xp.asnumpy(maps[2], stream=self.stream)

    def get_polygon_traversability(self, polygon, result):
        polygon = xp.asarray(polygon)
        area = calculate_area(polygon)
        pmin = self.center[:2] - self.map_length / 2 + self.resolution
        pmax = self.center[:2] + self.map_length / 2 - self.resolution
        polygon[:, 0] = polygon[:, 0].clip(pmin[0], pmax[0])
        polygon[:, 1] = polygon[:, 1].clip(pmin[1], pmax[1])
        polygon_min = polygon.min(axis=0)
        polygon_max = polygon.max(axis=0)
        polygon_bbox = cp.concatenate([polygon_min, polygon_max]).flatten()
        polygon_n = polygon.shape[0]
        clipped_area = calculate_area(polygon)
        self.polygon_mask_kernel(
            polygon,
            self.center[0],
            self.center[1],
            polygon_n,
            polygon_bbox,
            self.mask,
            size=(self.cell_n * self.cell_n),
        )
        masked, masked_isvalid = get_masked_traversability(self.elevation_map, self.mask)
        if masked_isvalid.sum() > 0:
            t = masked.sum() / masked_isvalid.sum()
        else:
            t = 0.0
        is_safe, un_polygon = is_traversable(
            masked, self.param.safe_thresh, self.param.safe_min_thresh, self.param.max_unsafe_n
        )
        untraversable_polygon_num = 0
        if un_polygon is not None:
            un_polygon = transform_to_map_position(un_polygon, self.center[:2], self.cell_n, self.resolution)
            untraversable_polygon_num = un_polygon.shape[0]
        if clipped_area < 0.001:
            is_safe = False
            print("requested polygon is outside of the map")
        result[...] = np.array([is_safe, t, area])
        self.untraversable_polygon = un_polygon
        return untraversable_polygon_num

    def get_untraversable_polygon(self, untraversable_polygon):
        untraversable_polygon[...] = xp.asnumpy(self.untraversable_polygon)

    def initialize_map(self, points, method="cubic"):
        self.clear()
        with self.map_lock:
            points = cp.asarray(points)
            indices = transform_to_map_index(points[:, :2], self.center[:2], self.cell_n, self.resolution)
            points[:, :2] = indices.astype(points.dtype)
            points[:, 2] -= self.center[2]
            self.map_initializer(self.elevation_map, points, method)
            if self.param.dilation_size_initialize > 0:
                for i in range(2):
                    self.dilation_filter_kernel_initializer(
                        self.elevation_map[0],
                        self.elevation_map[2],
                        self.elevation_map[0],
                        self.elevation_map[2],
                        size=(self.cell_n * self.cell_n),
                    )
            self.update_upper_bound_with_valid_elevation()


if __name__ == "__main__":
    #  Test script for profiling.
    #  $ python -m cProfile -o profile.stats elevation_mapping.py
    #  $ snakeviz profile.stats
    xp.random.seed(123)
    R = xp.random.rand(3, 3)
    t = xp.random.rand(3)
    print(R, t)
    param = Parameter(
        use_chainer=False, weight_file="../config/weights.dat", plugin_config_file="../config/plugin_config.yaml"
    )
    param.additional_layers =['feat_0','feat_1', 'rgb']
    param.fusion_algorithms = ['average','average','color']
    elevation = ElevationMap(param)
    layers = ["elevation", "variance", "traversability", "min_filter", "smooth", "inpaint","rgb"]
    points = xp.random.rand(100000, len(layers))

    channels = ['x','y','z']+ param.additional_layers
    print(channels)
    data = np.zeros((elevation.cell_n - 2, elevation.cell_n - 2), dtype=np.float32)
    for i in range(20):
        elevation.input(points,channels, R, t, 0, 0)
        elevation.update_normal(elevation.elevation_map[0])
        pos = np.array([i * 0.01, i * 0.02, i * 0.01])
        elevation.move_to(pos,R)
        for layer in layers:
            elevation.get_map_with_name_ref(layer, data)
        print(i)

import cupy as cp
import numpy as np
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase

from .PBA._distance_transform import distance_transform_edt


class SdfPlugin(PluginBase):
    def __init__(
        self,
        occupied_threshold: float = 0.3,
        **kwargs,
    ):
        """This is a filter to create a signed distance field from the traversability map.
        It makes use of the PBA+ algorithm to create the SDF.
        Copyright: (c) 2019 School of Computing, National University of Singapore

        Args:
            occupied_threshold (float): The threshold below which a cell is considered occupied.
        """
        super().__init__()
        self.occupied_threshold = float(occupied_threshold)
                # TODO: move to params
        self.occupancy_layer_name = "traversable_fusion_min_layer"

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map,
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            semantic_map (elevation_mapping_cupy.semantic_map.SemanticMap):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """


        occupancy_layer_ind = plugin_layer_names.index(self.occupancy_layer_name)
        occupancy_layer = plugin_layers[occupancy_layer_ind]
        inv_occupancy_grid = cp.where(occupancy_layer < self.occupied_threshold, 0, 1)
        
        resolution = semantic_map.param.resolution
        self.sdf = distance_transform_edt(inv_occupancy_grid,sampling=resolution)

        return self.sdf


        


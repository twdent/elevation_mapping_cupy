import cupy as cp
import numpy as np
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class TravFusion(PluginBase):
    def __init__(
        self,
        cell_n: int = 100,
        geo_trav_weight: float = 0.5,
        **kwargs,
    ):
        """This is a filter to create a weighted average of the geometric traversability map and the semantic map.

        Args:
            cell_n (int): width and height of the elevation map.
            geo_trav_weight (float): 0-1 weight of the geometric traversability map.
            **kwargs ():
        """
        super().__init__()
        self.traversable = cp.zeros((cell_n, cell_n), dtype=cp.float32)
        self.geo_trav_weight = float(geo_trav_weight) 
        

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
        #get the elevation_map[3] layer (traversability) and 
        #the semantic_map class probs layer - image currently? - so first channel for traversability
        # /traversable_segmentation/class_probabilities [sensor_msgs/Image]

        probs_layer_ind = semantic_map.layer_names.index('class_probabilities')
        sem_trav_probs = semantic_map.semantic_map[probs_layer_ind]

        geometric_trav_ind = layer_names.index('traversability')
        geometric_trav = elevation_map[geometric_trav_ind]
        
        self.traversable : cp.ndarray = self.geo_trav_weight* geometric_trav + (1-self.geo_trav_weight)*sem_trav_probs

        return self.traversable

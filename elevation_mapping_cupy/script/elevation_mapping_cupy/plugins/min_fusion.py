import cupy as cp
import cv2
import numpy as np
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class TravFusion(PluginBase):
    def __init__(
        self,
        cell_n: int = 100,
        geo_trav_weight: float = 0.5,
        min_safe_geo: float = 0.2,
        min_safe_sem: float = 0.1,
        **kwargs,
    ):
        """This is a filter to create a weighted average of the geometric traversability map and the semantic map.
        The min_safe_geo and min_safe_sem are the minimum values for the two maps to be considered safe.

        Args:
            cell_n (int): width and height of the elevation map.
            geo_trav_weight (float): 0-1 weight of the geometric traversability map.
            **kwargs ():
        """
        super().__init__()
        self.traversable = cp.zeros((cell_n, cell_n), dtype=cp.float32)
        self.geo_trav_weight = float(geo_trav_weight) 
        self.min_safe_geo = float(min_safe_geo)
        self.min_safe_sem = float(min_safe_sem)
        

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
        #this can be called before sematic map is created, so we need to check if the trav layer is present
        sem_trav_prob = self.get_layer_data(elevation_map, layer_names, plugin_layers, 
                                                plugin_layer_names, semantic_layers, semantic_layer_names,
                                                "trav")
        if sem_trav_prob is None:
            print("Couldn't find trav layer in semantic map")
            return self.traversable
        
        geometric_trav = self.get_layer_data(elevation_map, layer_names, plugin_layers, 
                                             plugin_layer_names, semantic_layers, semantic_layer_names,
                                             "traversability")
        
        weighted_sum = self.geo_trav_weight* geometric_trav + (1-self.geo_trav_weight)*sem_trav_prob

        # min fusion - min of the two traversability maps if less than min_safe for each map
        # else weighted sum of the two maps
        
        self.traversable = cp.where(geometric_trav < self.min_safe_geo, cp.minimum(geometric_trav, sem_trav_prob), 
                                    weighted_sum)
        self.traversable = cp.where(sem_trav_prob < self.min_safe_sem, cp.minimum(geometric_trav, sem_trav_prob), 
                                    weighted_sum)

        return self.traversable

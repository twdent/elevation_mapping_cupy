#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
from typing import List
import cupyx.scipy.ndimage as ndimage
import numpy as np
import cv2 as cv


from .plugin_manager import PluginBase


class Inpainting(PluginBase):
    """This is a filter to smoothen

    ...

    Attributes
    ----------
    cell_n: int
        width and height of the elevation map.
    """

    def __init__(self, cell_n: int = 100, method: str = "telea", input_layer_name: str = "elevation", **kwargs):
        super().__init__()
        self.input_layer_name = input_layer_name
        if method == "telea":
            self.method = cv.INPAINT_TELEA
        elif method == "ns":  # Navier-Stokes
            self.method = cv.INPAINT_NS
        else:  # default method
            self.method = cv.INPAINT_TELEA

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
    ) -> cp.ndarray:
        if self.input_layer_name in layer_names:
            idx = layer_names.index(self.input_layer_name)
            h = elevation_map[idx]
            mask = cp.asnumpy(cp.isnan(h).astype("uint8"))
        elif self.input_layer_name in plugin_layer_names:
            idx = plugin_layer_names.index(self.input_layer_name)
            h = plugin_layers[idx]
            mask = cp.asnumpy(cp.isnan(h).astype("uint8"))
        else:
            print("layer name {} was not found. Using elevation layer.".format(self.input_layer_name))
            h = elevation_map[0]
            mask = cp.asnumpy((elevation_map[2] < 0.5).astype("uint8"))
        if (mask < 1).any():
            h = np.nan_to_num(h)
            h_max = float(h.max())
            h_min = float(h.min())
            h = cp.asnumpy((h - h_min) * 255 / (h_max - h_min)).astype("uint8")
            dst = np.array(cv.inpaint(h, mask, 1, self.method))
            h_inpainted = dst.astype(np.float32) * (h_max - h_min) / 255.0 + h_min
            return cp.asarray(h_inpainted).astype(np.float64)
        else:
            return h

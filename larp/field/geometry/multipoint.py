from __future__ import annotations

import numpy as np

from larp.types import Point
from larp.field.geometry.geometry import MultiRGJGeometry

"""
Author: Josue N Rivera

x are assumed to be a list of point coordinates in euclidean space

"""


__all__ = ["MultiPointRGJ"]


class MultiPointRGJ(MultiRGJGeometry):
    RGJType = "MultiPoint"

    def __init__(self, coordinates: np.ndarray, repulsion:np.ndarray | None = None, **kwargs) -> None:
        super().__init__(coordinates=coordinates, repulsion=repulsion)
        self.bbox = self.coordinates.copy()

    def in_bbox(self, x:Point) -> bool:
        return any(self.bbox == x)

    def repulsion_vector(self, x: np.ndarray, min_dist_select:bool = True, **kwargs) -> np.ndarray:
        x = np.atleast_2d(x).astype(float)

        n = x.shape[0]
        m = self.coordinates.shape[0]

        x = np.tile(x, (m, 1, 1)).transpose(1, 0, 2)
        y = np.tile(self.coordinates, (n, 1, 1))
        diff = y - x

        if min_dist_select:
            matrix = self.get_dist_matrix(scaled=True, inverted=True)
            Adiff = np.matmul(diff, matrix)
            dist = (Adiff*diff).sum(-1)
            select = dist.argmin(1)
            diff = diff[np.arange(len(select)), select]
        else:
            diff = diff.swapaxes(0, 1)

        return diff
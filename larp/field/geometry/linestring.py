from __future__ import annotations

import numpy as np

from larp.field.geometry.geometry import MultiRGJGeometry

"""
Author: Josue N Rivera

x are assumed to be a list of point coordinates in euclidean space

"""


__all__ = ["LineStringRGJ"]


class LineStringRGJ(MultiRGJGeometry):
    RGJType = "LineString"

    def __init__(self, coordinates: np.ndarray, repulsion:np.ndarray | None = None, **kwargs) -> None:
        super().__init__(coordinates=coordinates, repulsion=repulsion, **kwargs)
        self.lines_n = len(self.coordinates) - 1
        self.points_in_line_pair = np.stack([self.coordinates[:-1], self.coordinates[1:]], axis=1)

    def set_coordinates(self, new_coords):
        super().set_coordinates(new_coords)
        self.lines_n = len(self.coordinates) - 1
        self.points_in_line_pair = np.stack([self.coordinates[:-1], self.coordinates[1:]], axis=1)
    
    def __repulsion_vector_one_line__(self, args) -> np.ndarray:
        x, line = args
        x2_d_x1 = line[1:2] - line[0:1]
        x_d_x1 = x - line[0:1]

        x12dotxx1 = (x2_d_x1*x_d_x1).sum(1, keepdims=True)
        x12dotx12 = (x2_d_x1*x2_d_x1).sum(1, keepdims=True)

        g = line[0] + np.clip(x12dotxx1/x12dotx12, 0.0, 1.0)*(x2_d_x1)
        return x - g
    
    def repulsion_vector(self, x: np.ndarray, min_dist_select:bool = True, **kwargs) -> np.ndarray:
        vectors:np.ndarray = [self.__repulsion_vector_one_line__((x, line)) for line in self.points_in_line_pair]

        vectors = np.stack(vectors, axis=0)

        if min_dist_select:
            vectors = vectors.swapaxes(0, 1)
            matrix = self.get_dist_matrix(scaled=True, inverted=True)
            nvectors = np.matmul(vectors, matrix)
            dist = (vectors*nvectors).sum(-1)
            select = dist.argmin(1)
            vectors = vectors[np.arange(len(select)), select]
        
        return vectors.reshape(-1, 2)
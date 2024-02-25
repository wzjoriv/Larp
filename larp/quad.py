from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
from larp import PotentialField

from larp.types import FieldSize, Point

"""
Author: Josue N Rivera
Date: 9/29/2023
Generate the quadtree and graph from the potential field
"""

def __list_to_dict__(array:Union[np.ndarray, List[float]]):
    array = np.array(array)
    values = np.unique(array)

    return dict(zip(values, np.arange(len(values))))

class QuadTree():

    def __init__(self, field: PotentialField,
                 minimum_sector_length:float = 5.0,
                 boundaries:Union[np.ndarray, List[float]] = np.arange(0.2, 0.8, 0.2),
                 size:Optional[float] = None,
                 build_tree:bool = False) -> None:
        
        self.field = field
        self.min_sector_size = minimum_sector_length
        self.size = size if size is not None else np.max(self.field.size)

        self.boundaries = np.sort(np.array(boundaries))[::-1]
        self.n_zones = len(self.boundaries) + 1
        self.__zones_rad_ln = -np.log(self.boundaries)
        self.ZONEToMaxRANGE = np.concatenate([[1.0, 1.0], self.boundaries])
        self.ZONEToMinRANGE = np.concatenate([self.boundaries[0:1], self.boundaries, [0.0]])

        self.root = None
        self.leaves:List[QuadNode] = []

        if build_tree:
            self.build()

    def mark_leaf(self, quad:QuadNode) -> None:

        self.leaves.append(quad)
        quad.leaf = True

    def __approximated_PF_zones__(self, center_point:Point, size:float, filter_idx:Optional[List[int]] = None) -> Tuple[List[int], np.ndarray]: 
        n_rgjs = len(filter_idx)
        zones = np.ones(n_rgjs, dtype=int) * self.n_zones

        dist_sqr = self.field.squared_dist_list([center_point], filted_idx=filter_idx, scaled=False).ravel()
        zone0_select = dist_sqr <= (size*size)/2.0
        zones[zone0_select] = 0

        if sum(zone0_select) < n_rgjs:
            rgjs_idx = filter_idx[~zone0_select]
            vectors, idxs = self.field.repulsion_vectors([center_point], filted_idx=rgjs_idx, reference_idx=True).reshape(-1, 2)
            uni_vectors = vectors/np.linalg.norm(vectors, axis=1, keepdims=True)
            # TODO: Use idxs to get minimum dist per rgj object

            dist_sqr = self.field.squared_dist(center_point - uni_vectors*size/np.sqrt(2), filted_idx=rgjs_idx).ravel()
            dist_sqr_bins = self.__zones_rad_ln

            zones[~zone0_select] = np.digitize(dist_sqr, dist_sqr_bins, right=True) + 1

        return zones

    def build(self) -> Optional[QuadNode]:
        self.leaves:List[QuadNode] = []

        def dfs(center_point:Point, size:float, filter_idx:Optional[List[int]]) -> QuadNode:
            quad = QuadNode(center_point=center_point, size=size)

            zones = self.__approximated_PF_zones__(center_point=center_point, size=size, filter_idx=filter_idx)
            quad.boundary_zone = zone = min(zones)
            quad.boundary_max_range = self.ZONEToMaxRANGE[zone]
            
            if size <= self.min_sector_size or zone == self.n_zones:
                # stop subdividing if size is too small or the active zones are too far away
                self.mark_leaf(quad)
                return quad
            
            size2 = size/2.0
            if zone > 0:
                # stop subdiving if sphere does not leave zone
                lower_range = self.ZONEToMinRANGE[zone]
                rgjs_idx = filter_idx[zones == zone]

                vectors = self.field.repulsion_vectors([center_point], filted_idx=rgjs_idx).reshape(-1, 2)
                uni_vectors = vectors/np.linalg.norm(vectors, axis=1, keepdims=True)

                if (self.field.eval(center_point + uni_vectors*size/np.sqrt(2)) >= lower_range).any():
                    self.mark_leaf(quad)
                    return quad

            size4 = size2/2.0
            new_filter_idx = filter_idx[zones < self.n_zones]
            quad['tl'] = dfs(center_point + np.array([-size4, size4]), size2, new_filter_idx)
            quad['tr'] = dfs(center_point + np.array([ size4, size4]), size2, new_filter_idx)
            quad['bl'] = dfs(center_point + np.array([-size4,-size4]), size2, new_filter_idx)
            quad['br'] = dfs(center_point + np.array([ size4,-size4]), size2, new_filter_idx)

            return quad
        
        self.root = dfs(self.field.center_point, self.size, np.arange(len(self.field)))
        return self.root
    
    def to_boundary_lines_collection(self, margin=0.1):
        lines = [quad.to_boundary_lines(margin=margin) for quad in self.leaves]
        
        return [path for line in lines for path in line]
    
    def get_quad_maximum_range(self):
        return np.array([quad.boundary_max_range for quad in self.leaves])

class QuadNode():

    chdToIdx = __list_to_dict__(['tl', 'tr', 'bl', 'br'])
    nghToIdx = __list_to_dict__(['tl', 't', 'tr', 'r', 'br', 'b', 'bl', 'l'])
    
    def __init__(self, center_point:Point, size:float) -> None:
        self.center_point = center_point
        self.size = size
        self.leaf = False
        self.boundary_zone:int = 0
        self.boundary_max_range:float = 1.0

        self.children = [None]*len(self.chdToIdx)
        self.neighbors = [None]*len(self.nghToIdx)

    def __getitem__(self, idx:Union[str, int, tuple, list]) -> Union[QuadNode, List[QuadNode]]:

        if isinstance(idx, (list, tuple)):
            n = len(idx)
            out = [None]*n

            for i in range(n):
                id = self.chdToIdx[idx[i]] if not isinstance(idx[i], int) else idx[i]
                out[i] = self.neighbors[id]
            return out
        
        else:
            idx = self.chdToIdx[idx] if not isinstance(idx, int) else idx
            return self.children[idx]

    def __setitem__(self, idx:Union[str, int, tuple, list], value:QuadNode) -> None:
        if isinstance(idx, (list, tuple)):
            for id in idx:
                id = self.chdToIdx[id] if not isinstance(id, int) else id
                self.neighbors[id] = value

        else:
            idx = self.chdToIdx[idx] if not isinstance(idx, int) else idx
            self.children[idx] = value
    
    def to_boundary_lines(self, margin=0.1) -> Tuple[np.ndarray, np.ndarray]:
        size2 = self.size/2.0
        offset = np.array([
            [-1.0, 1.0],
            [ 1.0, 1.0],
            [ 1.0,-1.0],
            [-1.0,-1.0],
            [-1.0, 1.0],
        ]) * size2
        offset[offset > 0] = offset[offset > 0] - margin
        offset[offset <= 0] = offset[offset <= 0] + margin
        path = self.center_point + offset

        return path[:, 0], path[:, 1]



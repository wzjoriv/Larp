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
                 size:Optional[float] = None) -> None:
        
        self.field = field
        self.min_sector_size = minimum_sector_length
        self.size = size if size is not None else np.min(self.field.size)

        self.boundaries = np.sort(np.array(boundaries))[::-1]
        self.ZONEToMaxRANGE = np.append([1.0, 1.0], self.boundaries)
        self.ZONEToMinRANGE = np.append(self.boundaries[0:1], self.boundaries, [0.0])
        self.__zones_rad_ln = -np.log(np.append([1.0], self.boundaries))
        self.n_zones = len(self.boundaries) + 1

        self.root = None
        self.leaves = []

    def __approximated_PF_zones__(self, center_point:Point, size:float, filter_idx:Optional[List[int]] = None) -> Tuple[List[int], np.ndarray]: 

        dist_sqr = self.field.squared_dist_list([center_point], filted_idx=filter_idx).ravel()
        dist_sqr_bins = (size*size)/2.0 + np.sqrt(2)*size*np.sqrt(self.__zones_rad_ln) + self.__zones_rad_ln

        zones = np.digitize(dist_sqr, dist_sqr_bins, right=True)

        return zones, dist_sqr

    def build(self) -> Optional[QuadNode]:

        def dfs(center_point:Point, size:float, filter_idx:Optional[List[int]]) -> QuadNode:
            quad = QuadNode(center_point=center_point, size=size)

            zones, dist_sqr = self.__approximated_PF_zones__(center_point=center_point, size=size, filter_idx=filter_idx)
            quad.boundary_zone = zone = min(zones)
            quad.boundary_max_range = self.ZONEToMaxRANGE[zone]
            
            size2 = size/2.0
            if size2 <= self.min_sector_size or zone == self.n_zones:
                # stop subdividing if next size is too small or the active zones are too far away or range cannot be passed
                self.leaves.append(quad)
                return quad
            
            upper_range, lower_range = self.ZONEToMaxRANGE[zone], self.ZONEToMinRANGE[zone]
            
            # TODO: 
            # 1) Check if other range can exist within size boundary. If not, stop subdividing

            ### Attempt 1: Will likely not work
            if 2*size**2 < -np.log(upper_range) - np.log(lower_range) - np.sqrt(np.log(upper_range)*np.log(lower_range)):
                # stop subdividing if size is small enough 

                self.leaves.append(quad)
                return quad
            
            ### Atempt 2: check for unscaled distance inside circle (zone 0), and scaled outside (zone 1+)
            # -----

            size4 = size2/2.0
            new_filter_idx = filter_idx[zones < self.n_zones]
            quad['tl'] = dfs(center_point + np.array([-size4, size4]), size2, new_filter_idx)
            quad['tr'] = dfs(center_point + np.array([ size4, size4]), size2, new_filter_idx)
            quad['bl'] = dfs(center_point + np.array([-size4,-size4]), size2, new_filter_idx)
            quad['br'] = dfs(center_point + np.array([ size4,-size4]), size2, new_filter_idx)

            return quad
        
        self.root = dfs(self.field.center_point, self.size, np.arange(len(self.field)))
        return self.root
    
    def to_line_collection(self):
        # TODO: return segmented lines for matplotlib's LineCollection
        pass 


class QuadNode():

    chdToIdx = __list_to_dict__(['tl', 'tr', 'bl', 'br'])
    nghToIdx = __list_to_dict__(['tl', 't', 'tr', 'r', 'br', 'b', 'bl', 'l'])
    
    def __init__(self, center_point:Point, size:float) -> None:
        self.center_point = center_point
        self.size = size
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



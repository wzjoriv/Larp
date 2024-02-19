from __future__ import annotations
from typing import List, Optional, Union
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
        self.min_sector_n = minimum_sector_length
        self.boundaries = np.sort(np.array(boundaries))[::-1]
        self.size = size if size is not None else np.min(self.field.size)

        self.root = None

    def __approximated_PF_bins__(self, center_point:Point, size:float, filter_idx:Optional[List[int]] = None):
        # TODO: Filter and set bins

        dist = self.field.squared_dist_list([center_point], filted_idx=filter_idx).ravel()

        bins =  []
        return bins

    def build(self) -> Optional[QuadNode]:

        def dfs(center_point:Point, size:float, filter_idx:Optional[List[int]]) -> QuadNode:
            n = size/2.0
            quad = QuadNode(center_point=center_point, size=size)

            if n <= self.min_sector_n: # stop subdividing if next size is too small 
                return quad
            
            bins = self.__approximated_PF_bins__(center_point=center_point, size=size, filter_idx=filter_idx)
            
            n2 = n/2.0
            quad['tl'] = dfs(center_point + np.array([-n2, n2]), n)
            quad['tr'] = dfs(center_point + np.array([ n2, n2]), n)
            quad['bl'] = dfs(center_point + np.array([-n2,-n2]), n)
            quad['br'] = dfs(center_point + np.array([ n2,-n2]), n)

            return quad
        
        self.root = dfs(self.field.center_point, self.size, None)
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
        self.boundary_bins = []

        self.children = [None]*len(self.chdToIdx)
        self.neighbors = [None]*len(self.nghToIdx)

    def __getitem__(self, idx:Union[str, int, tuple]) -> Union[QuadNode, List[QuadNode]]:

        if isinstance(idx, tuple):
            n = len(idx)
            out = [None]*n

            for i in range(n):
                id = self.chdToIdx[idx[i]] if not isinstance(idx[i], int) else idx[i]
                out[i] = self.neighbors[id]
            return out
        
        else:
            idx = self.chdToIdx[idx] if not isinstance(idx, int) else idx
            return self.children[idx]

    def __setitem__(self, idx:Union[str, int], value:QuadNode) -> None:
        if isinstance(idx, tuple):
            for id in idx:
                id = self.chdToIdx[id] if not isinstance(id, int) else id
                self.neighbors[id] = value

        else:
            idx = self.chdToIdx[idx] if not isinstance(idx, int) else idx
            self.children[idx] = value



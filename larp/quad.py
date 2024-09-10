from __future__ import annotations
from typing import List, Optional, Set, Tuple, Union
import numpy as np
from larp import PotentialField

from larp.types import Point

"""
Author: Josue N Rivera
Generate the quadtree from the potential field
"""

def __list_to_dict__(array:Union[np.ndarray, List[float]]):
    array = np.array(array)
    values = np.unique(array)

    return dict(zip(values, np.arange(len(values))))

class QuadTree():

    def __init__(self, field: PotentialField,
                 minimum_length_limit:float = 5.0,
                 maximum_length_limit:float = np.Inf,
                 edge_bounds:Union[np.ndarray, List[float]] = np.arange(0.2, 0.8, 0.2),
                 size:Optional[float] = None,
                 conservative:bool = False,
                 build_tree:bool = False) -> None:
        
        self.field = field
        self.min_sector_size = minimum_length_limit
        self.max_sector_size = maximum_length_limit
        self.size = size if size is not None else np.max(self.field.size)

        self.edge_bounds = np.sort(np.array(edge_bounds))[::-1]
        self.n_zones = len(self.edge_bounds) + 1
        self.__zones_rad_ln = -np.log(self.edge_bounds)
        self.ZONEToMaxRANGE = np.concatenate([[1.0, 1.0], self.edge_bounds])
        self.ZONEToMinRANGE = np.concatenate([self.edge_bounds[0:1], self.edge_bounds, [0.0]])
        self.conservative = conservative

        self.root = None
        self.leaves:Set[QuadNode] = set()

        if build_tree:
            self.build()

    def mark_leaf(self, quad:QuadNode) -> None:
        quad.leaf = True
        self.leaves.add(quad)

    def __approximated_PF_zones__(self, center_point:Point, size:float, filter_idx:Optional[List[int]] = None) -> Tuple[List[int], np.ndarray]: 
        n_rgjs = len(filter_idx)
        zones = np.ones(n_rgjs, dtype=int) * self.n_zones

        rep_vectors, refs_idxs = self.field.repulsion_vectors([center_point], filted_idx=filter_idx, min_dist_select=True, reference_idx=True)

        dist_sqr = (rep_vectors*rep_vectors).sum(1)
        zone0_select = dist_sqr <= (size*size)/2.0
        zones[zone0_select] = 0

        if sum(zone0_select) < n_rgjs:
            not_zone0_select = ~zone0_select
            rgjs_idx = filter_idx[not_zone0_select]
            vectors = rep_vectors[not_zone0_select]

            vectors = vectors.reshape(-1, 2)
            uni_vectors = vectors/np.linalg.norm(vectors, axis=1, keepdims=True)

            # TODO: Ensure squared distance is per rgj
            dist_sqr = self.field.squared_dist_per(center_point - uni_vectors*(size/np.sqrt(2)), idxs=rgjs_idx).ravel()

            zones[not_zone0_select] = np.digitize(dist_sqr, self.__zones_rad_ln, right=True) + 1

        return zones, rep_vectors, refs_idxs
    
    def __build__(self, center_point:Point, size:float, filter_idx:np.ndarray) -> QuadNode:
         
        quad = QuadNode(center_point=center_point, size=size)
        filter_n = len(filter_idx)

        if filter_n:
            zones, rep_vectors, refs_idxs = self.__approximated_PF_zones__(center_point=center_point, size=size, filter_idx=filter_idx)
            quad.boundary_zone = min(zones)
            
            select = zones < self.n_zones
            quad.rgj_idx = filter_idx[select]
            quad.rgj_zones = zones[select]
        else:
            quad.boundary_zone = self.n_zones

        quad.boundary_max_range = self.ZONEToMaxRANGE[quad.boundary_zone]
        
        size2 = size/2.0
        if size <= self.max_sector_size:
            if size2 < self.min_sector_size or quad.boundary_zone == self.n_zones:
                # stop subdividing if size is too small or the active zones are too far away
                self.mark_leaf(quad)
                return quad
            if self.conservative and quad.boundary_zone > 0:
                # stop subdiving if sphere does not leave zone
                lower_range = self.ZONEToMinRANGE[quad.boundary_zone]

                select = zones == quad.boundary_zone
                vectors, refs_idxs = rep_vectors[select], refs_idxs[select]
                vectors = vectors.reshape(-1, 2)
                uni_vectors = vectors/np.linalg.norm(vectors, axis=1, keepdims=True)

                bounds_evals = self.field.eval_per(center_point + uni_vectors*(size/np.sqrt(2)), idxs=refs_idxs)
                if (bounds_evals >= lower_range).any():
                    self.mark_leaf(quad)
                    return quad

        size4 = size2/2.0
        quad['tl'] = self.__build__(center_point + np.array([-size4, size4]), size2, quad.rgj_idx)
        quad['tr'] = self.__build__(center_point + np.array([ size4, size4]), size2, quad.rgj_idx)
        quad['bl'] = self.__build__(center_point + np.array([-size4,-size4]), size2, quad.rgj_idx)
        quad['br'] = self.__build__(center_point + np.array([ size4,-size4]), size2, quad.rgj_idx)

        return quad

    def build(self) -> QuadNode:
        self.leaves:Set[QuadNode] = set()
        
        self.root = self.__build__(self.field.center_point, self.size, np.arange(len(self.field)))
        return self.root
    
    def to_boundary_lines_collection(self, margin=0.1) -> List[np.ndarray]:
        lines = [quad.to_boundary_lines(margin=margin) for quad in self.leaves]
        
        return [path for line in lines for path in line]
    
    def get_quad_maximum_range(self) -> np.ndarray:
        return np.array([quad.boundary_max_range for quad in self.leaves])
    
    def find_quads(self, x:Union[List[Point],np.ndarray]) -> List[QuadNode]:
        """ Finds quad for given points

        * Pool parallization not possible because quad memory reference is needed
        """
        x = np.array(x)

        def subdivide(x:Point, quad:QuadNode) -> List[QuadNode]:
            if quad is None or quad.leaf:
                return quad

            direction = x - quad.center_point
            if direction[1] >= 0.0:
                quadstr = "tr" if direction[0] >= 0.0 else "tl"
            else:
                quadstr = "br" if direction[0] >= 0.0 else "bl"

            return subdivide(x, quad=quad[quadstr])

        return [subdivide(x=xi, quad=self.root) for xi in x]
    
    def __search_leaves__(self, quad:QuadNode):
        if quad is None: raise TypeError(f"Branch missing leaf for quad {str(quad)}")
        if quad.leaf: return [quad]

        out = []
        for child in quad.children:
            out.extend(self.__search_leaves__(child))

        return out

    def search_leaves(self, quad:Optional[QuadNode] = None) -> Set[QuadNode]:
        quad = self.root if quad is None else quad
        return set(self.__search_leaves__(quad))
    
    def get_quad_zones(self):
        return np.array([quad.boundary_zone for quad in self.leaves], dtype=int)
    
    def toDict(self):
        def __save_quad__(quad:Optional[QuadNode]) -> dict:

            if quad is None:
                return None
            
            data = {
                'center_point': quad.center_point,
                'size': quad.size,
                'leaf': quad.leaf,
                'boundary_zone': quad.boundary_zone,
                'boundary_max_range': quad.boundary_max_range,
                'rgj_idx': quad.rgj_idx,
                'rgj_zones': quad.rgj_zones,
                'children': [__save_quad__(child) for child in quad.children]
            }

            return data

        data = {
            'field': self.field.toRGeoJSON(),
            'min_sector_size': self.min_sector_size,
            'max_sector_size': self.max_sector_size,
            'size': self.size,
            'edge_bounds': self.edge_bounds,
            'n_zones': self.n_zones,
            '__zones_rad_ln': self.__zones_rad_ln,
            'ZONEToMaxRANGE': self.ZONEToMaxRANGE,
            'ZONEToMinRANGE': self.ZONEToMinRANGE,
            'conservative': self.conservative,
            'root': __save_quad__(self.root)
        }

        return data

    def fromDict(self, data:dict):

        def __load_quad__(quad_data:Optional[dict]) -> Optional[QuadNode]:

            if quad_data is None:
                return None
            
            quad = QuadNode(center_point = quad_data['center_point'],
                            size = quad_data['size'])
            
            quad.leaf = quad_data['leaf']
            quad.boundary_zone = quad_data['boundary_zone']
            quad.boundary_max_range = quad_data['boundary_max_range']
            quad.rgj_idx = quad_data['rgj_idx']
            quad.rgj_zones = quad_data['rgj_zones']
            quad.children = [__load_quad__(child) for child in quad_data['children']]

            return quad

        self.min_sector_size = data['min_sector_size'] 
        self.max_sector_size = data['max_sector_size'] 
        self.size = data['size'] 
        self.edge_bounds = data['edge_bounds'] 
        self.n_zones = data['n_zones'] 
        self.__zones_rad_ln = data['__zones_rad_ln'] 
        self.ZONEToMaxRANGE = data['ZONEToMaxRANGE'] 
        self.ZONEToMinRANGE = data['ZONEToMinRANGE']
        self.root = __load_quad__(data['root'])
        self.leaves = self.search_leaves()

    def quad_to_image(self, quad:Optional[QuadNode] = None, resolution:int = 200, margin:float = 0.0) -> np.ndarray:

        if quad is None:
            quad = self.root

        return self.field.to_image(resolution=resolution,
                                   margin=margin,
                                   center_point=quad.center_point,
                                   size=[quad.size]*2,
                                   filted_idx=quad.rgj_idx)

class QuadNode():

    chdToIdx = __list_to_dict__(['tl', 'tr', 'bl', 'br'])
    nghToIdx = __list_to_dict__(['tl', 't', 'tr', 'r', 'br', 'b', 'bl', 'l'])
    
    def __init__(self, center_point:Point, size:float) -> None:
        self.center_point = np.array(center_point)
        self.size = size
        self.leaf = False
        self.boundary_zone:int = 0
        self.boundary_max_range:float = 1.0

        self.rgj_idx = np.array([], dtype=int)
        self.rgj_zones = np.array([], dtype=int)

        self.children = [None]*len(self.chdToIdx)
        self.neighbors = [None]*len(self.nghToIdx)

    def __getitem__(self, idx:Union[str, int, tuple, list]) -> Union[QuadNode, List[QuadNode]]:

        """
        If list or tuple given, then neighbors considered. Else, children will be considered.
        """

        if isinstance(idx, (list, tuple)):
            n = len(idx)
            out = [None]*n

            for i in range(n):
                id = self.nghToIdx[idx[i]] if not isinstance(idx[i], int) else idx[i]
                out[i] = self.neighbors[id]
            return out
        
        else:
            idx = self.chdToIdx[idx] if not isinstance(idx, int) else idx
            return self.children[idx]

    def __setitem__(self, idx:Union[str, int, tuple, list], value:QuadNode) -> None:
        if isinstance(idx, (list, tuple)):
            for id in idx:
                id = self.nghToIdx[id] if not isinstance(id, int) else id
                self.neighbors[id] = value
        else:
            idx = self.chdToIdx[idx] if not isinstance(idx, int) else idx
            self.children[idx] = value

    def __lt__(self, other:QuadNode):
        return self.boundary_max_range < other.boundary_max_range
    
    def to_boundary_lines(self, margin=0.1) -> Tuple[np.ndarray, np.ndarray]:
        size2 = self.size/2.0 - margin
        offset = np.array([
            [-1.0, 1.0],
            [ 1.0, 1.0],
            [ 1.0,-1.0],
            [-1.0,-1.0],
            [-1.0, 1.0],
        ]) * size2
        path = self.center_point + offset

        return path[:, 0], path[:, 1]
    
    def __str__(self) -> str:
        return f"Qd({self.center_point.tolist()}, {self.size})"



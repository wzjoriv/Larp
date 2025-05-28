from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings
from collections import defaultdict

import numpy as np
from larp import PotentialField

import larp.fn as lpf
from larp.field import RGJGeometry
from larp.types import Scaler, FieldSize, Point, RGJDict

"""
Author: Josue N Rivera
Generate the quadtree from the potential field
"""

def __make_index_map__(array: Union[np.ndarray, List]) -> Dict:
    """
    Creates a dictionary mapping each unique item in the input to a unique integer index,
    preserving the order of first appearance.

    Args:
        array (Union[np.ndarray, List]): A list or array of hashable items.

    Returns:
        Dict: A mapping from unique item → index.
    """
    array = list(array)
    unique_values = list(dict.fromkeys(array))
    return dict(zip(unique_values, range(len(unique_values))))

def __deduplicate_with_index_map__(array: List) -> Tuple[List, List[int]]:

    """
    Deduplicates a list while preserving order and provides an index map
    from the original list to the unique items list.

    Useful for compressing data and remapping computations onto a reduced set
    of unique values (e.g., quads or identifiers).

    Args:
        array (List): A list of hashable items.

    Returns:
        Tuple[List, List[int]]:
            - unique_items: List of unique items in first-seen order.
            - index_map: A list of the same length as the input, where each entry
                         maps to the corresponding index in `unique_items`.
    """
    if len(array) == len(set(array)):
        # All items are already unique — identity mapping
        return array, list(range(len(array)))

    item_to_index = {}
    unique_items = []
    index_map = []

    for item in array:
        if item not in item_to_index:
            item_to_index[item] = len(unique_items)
            unique_items.append(item)
        index_map.append(item_to_index[item])

    return unique_items, index_map

class QuadTree():

    def __init__(self, field: PotentialField,
                 minimum_length_limit:float = 5.0,
                 maximum_length_limit:float = np.inf,
                 edge_bounds:Union[np.ndarray, List[float]] = np.arange(0.2, 0.8, 0.2),
                 size:Optional[float] = None,
                 conservative:bool = False,
                 build_tree:bool = True) -> None:
        
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

    def __iter__(self):
        self.quad_idx = 0
        self.leaves_list = list(self.leaves)
        return self
    
    def __next__(self):
        if self.quad_idx >= len(self):
            raise StopIteration
        
        out = self.leaves_list[self.quad_idx]
        self.quad_idx += 1
        return out

    def __len__(self)->int:
        return len(self.leaves)

    def mark_leaf(self, quad:QuadNode) -> None:
        quad.leaf = True
        self.leaves.add(quad)

    def __approximated_PF_zones__(self, center_point:Point, size:float, filter_idx:Optional[List[int]] = None) -> Tuple[List[int], np.ndarray]: 
        n_rgjs = len(filter_idx)
        zones = np.ones(n_rgjs, dtype=int) * self.n_zones

        rep_vectors, refs_idxs = self.field.repulsion_vectors([center_point], filted_idx=filter_idx, min_dist_select=True, return_reference=True)

        dist_sqr = (rep_vectors*rep_vectors).sum(1)
        zone0_select = dist_sqr <= (size*size)/2.0
        zones[zone0_select] = 0

        if sum(zone0_select) < n_rgjs:
            not_zone0_select = ~zone0_select
            rgjs_idx = filter_idx[not_zone0_select]
            vectors = rep_vectors[not_zone0_select]

            vectors = vectors.reshape(-1, 2)
            uni_vectors = vectors/np.linalg.norm(vectors, axis=1, keepdims=True)

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
            if size2 <= self.min_sector_size or quad.boundary_zone == self.n_zones:
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

    def replace_branch(self, rootquad:QuadNode, child:str, branch:QuadNode):
            
            if rootquad[child] is not None:
                self.leaves -= self.search_leaves(rootquad[child])

            rootquad[child] = branch
            new_leaves = self.search_leaves(rootquad[child])
            self.leaves.update(new_leaves)

    def get_quad_maximum_range(self) -> np.ndarray:
        return np.array([quad.boundary_max_range for quad in self.leaves])

    def find_quad(self, x: Union[List['Point'], np.ndarray], max_depth: int = 1000) -> List['QuadNode']:
        """
        Efficiently finds the quad node for each point, minimizing redundant traversal.

        Args:
            x (List[Point] or np.ndarray): List of 2D points.
            max_depth (int): Maximum depth to search in the quad tree.

        Returns:
            List[QuadNode]: Quad node for each point, in input order.
        """

        x = np.atleast_2d(x).astype(float)

        n_points = len(x)
        results = [None] * n_points

        def batch_traverse(quad: 'QuadNode', point_indices: np.ndarray, depth: int):
            if quad is None or quad.leaf or depth >= max_depth:
                for idx in point_indices:
                    results[idx] = quad
                return

            # Group point indices by quadrant
            children = defaultdict(list)
            for idx in point_indices:
                dx, dy = x[idx] - quad.center_point
                if dy >= 0.0:
                    direction = 'tr' if dx >= 0.0 else 'tl'
                else:
                    direction = 'br' if dx >= 0.0 else 'bl'
                children[direction].append(idx)

            # Recurse into each child with only relevant points
            for direction, indices in children.items():
                batch_traverse(quad[direction], np.array(indices), depth + 1)

        batch_traverse(self.root, np.arange(n_points), depth=0)
        return results
    
    def find_quads_chain(self, x: Union[List['Point'], np.ndarray], max_depth: int = 1000) -> List[List['QuadNode']]:
        """
        Efficiently finds the full quad traversal chain (from root to final quad) for each point.

        Args:
            x (List[Point] or np.ndarray): List of 2D points.
            max_depth (int): Maximum depth to search in the quad tree.

        Returns:
            List[List[QuadNode]]: A list of quad chains, one per point.
        """
        x = np.atleast_2d(x).astype(float)
        n_points = len(x)
        results = [[] for _ in range(n_points)]  # chain for each point

        def batch_traverse(quad: 'QuadNode', point_indices: np.ndarray, depth: int):
            if quad is None or quad.leaf or depth >= max_depth:
                for idx in point_indices:
                    results[idx].append(quad)
                return

            # Append current quad to each point's chain
            for idx in point_indices:
                results[idx].append(quad)

            # Group points by which direction they go in
            children = defaultdict(list)
            for idx in point_indices:
                dx, dy = x[idx] - quad.center_point
                if dy >= 0.0:
                    direction = 'tr' if dx >= 0.0 else 'tl'
                else:
                    direction = 'br' if dx >= 0.0 else 'bl'
                children[direction].append(idx)

            # Recurse into each quadrant
            for direction, indices in children.items():
                batch_traverse(quad[direction], np.array(indices), depth + 1)

        batch_traverse(self.root, np.arange(n_points), depth=0)
        return results
    
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

    def to_image(self, return_potential=False, return_extent: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, List[float]]]:
        """
        Render a top-down raster image of the quadtree zoning layout.

        Args:
            return_potential (bool): Whether to return the potential limit of each zone.
            return_extent (bool): Whether to return the real-world coordinate extent.

        Returns:
            image (np.ndarray): 2D array with potential or zone values of sectors.
            extent (Optional[List[float]]): [xmin, xmax, ymin, ymax] real-world bounds if return_extent is True.
        """
        # Get resolution as a power of two
        resolution = 2 ** int(np.floor(np.log2(self.root.size / self.min_sector_size)))+1
        pixel_size = self.root.size / resolution

        image = np.ones((resolution, resolution), dtype=int) * self.n_zones

        half_size = self.root.size / 2.0
        lower_bound = self.root.center_point - half_size
        upper_bound = self.root.center_point + half_size

        for quad in self.leaves:
            if quad.boundary_zone == self.n_zones:
                continue

            quad_half = quad.size / 2.0
            x0 = int((quad.center_point[0] - quad_half - lower_bound[0]) / pixel_size)
            y0 = int((upper_bound[1] - (quad.center_point[1] + quad_half)) / pixel_size)
            block_size = max(1, int(np.floor(quad.size / pixel_size)))

            # Ensure bounds are within the image dimensions
            x1 = min(x0 + block_size, resolution)
            y1 = min(y0 + block_size, resolution)

            image[y0:y1, x0:x1] = quad.boundary_zone

        if return_potential:
            image = self.ZONEToMaxRANGE[image]

        if return_extent:
            return image, [lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1]]
        return image


    def quad_to_image(self, quad:Optional[QuadNode] = None, resolution:int = 200, margin:float = 0.0) -> np.ndarray:

        if quad is None:
            quad = self.root

        return self.field.to_image(resolution=resolution,
                                   margin=margin,
                                   center_point=quad.center_point,
                                   size=[quad.size]*2,
                                   filted_idx=quad.rgj_idx)

class QuadNode():

    chdToIdx = __make_index_map__(['tl', 'tr', 'bl', 'br'])
    nghToIdx = __make_index_map__(['tl', 't', 'tr', 'r', 'br', 'b', 'bl', 'l'])
    
    def __init__(self, center_point:Point, size:float) -> None:
        self.center_point = np.atleast_1d(center_point).astype(float)
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
    
    def get_boundaries(self) -> np.ndarray:
        """
        Returns (xmin, ymin, xmax, ymax) boundaries of the quad.
        """
        cx, cy = self.center_point
        half = self.size / 2.0
        return np.array([cx - half, cy - half, cx + half, cy + half])
    
    def get_shared_edge(self, neighbor: 'QuadNode'):
        """
        Computes the shared boundary segment between this quad and a neighboring quad.

        Returns:
            np.ndarray: A (2, 2) array representing the endpoints of the shared edge.
                - If the quads share an edge (horizontal or vertical), the endpoints of that edge are returned.
                - If the quads only touch at a corner (diagonal adjacency), the single shared point is returned twice.
                - If the quads do not touch, returns None.

        Note:
        """
        if neighbor is None:
            return None

        b1 = self.get_boundaries()   # [xmin, ymin, xmax, ymax]
        b2 = neighbor.get_boundaries()

        # Extract boundaries
        x0 = max(b1[0], b2[0])  # max(xmin1, xmin2)
        y0 = max(b1[1], b2[1])  # max(ymin1, ymin2)
        x1 = min(b1[2], b2[2])  # min(xmax1, xmax2)
        y1 = min(b1[3], b2[3])  # min(ymax1, ymax2)

        # Check for overlap in both axes
        if not np.isclose(x0, x1) and not np.isclose(y0, y1):
            return None  # No overlap

        # Otherwise, it's a shared edge
        return np.array([[x0, y0], [x1, y1]]) # [xmin, ymin, xmax, ymax]

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

class QPotentailField(PotentialField):

    """ Potential Field class optimized by quadtree and to maintain them linked
    
    - Need all RGJ to be inside the quad tree area to be efficient and reliable
    """

    def __init__(self, field_quadtree:Union[PotentialField, QuadTree]):

        if isinstance(field_quadtree, PotentialField):
            field = field_quadtree
            quadtree = QuadTree(field, minimum_length_limit=np.max(field.size)/8, build_tree=True)
        else:
            field = field_quadtree.field
            quadtree = field_quadtree

        self.field = field
        self.quadtree = quadtree
        self.quadtree.conservative = False

    def __iter__(self):
        return self.field.__iter__()
    
    def __next__(self):
        return self.field.__next__()

    def __len__(self)->int:
        return len(self.field)
    
    def __group_points_by_quads_with_rgjs(self, points: np.ndarray, max_depth: int = 2):
        """
        Assign points to the deepest quad containing RGJs and group indices per quad.

        Args:
            points (np.ndarray): Array of points, shape (N, 2).
            max_depth (int): Maximum quadtree depth to traverse.

        Returns:
            unique_quads (List[QuadNode]): Unique quads with RGJs.
            quad_to_point_indices (Dict[int, List[int]]): Map from quad idx to list of point indices.
        """

        points = np.atleast_2d(points).astype(float)
        quad_chains = self.quadtree.find_quads_chain(points, max_depth=max_depth)

        final_quads = []
        for chain in quad_chains:
            for node in reversed(chain):
                if len(node.rgj_idx):
                    final_quads.append(node)
                    break
            else:
                final_quads.append(chain[0])  # fallback if no RGJs found

        unique_quads, point_to_quad_idx = __deduplicate_with_index_map__(final_quads)
        quad_to_point_indices = defaultdict(list)
        for pt_idx, quad_idx in enumerate(point_to_quad_idx):
            quad_to_point_indices[quad_idx].append(pt_idx)

        return unique_quads, quad_to_point_indices
    
    def set_all_repulsion(self, new_repulsion):
        self.field.set_all_repulsion(new_repulsion=new_repulsion)
        self.quadtree.build()

    def reload_bbox(self):
        self.field.reload_bbox()
    
    def reload_center_point(self, toggle=True, recal_size=False) -> Point:
        return self.field.reload_center_point(toggle=toggle, recal_size=recal_size)
    
    def get_extent(self, margin:float = 0.0) -> List[float]:
        return self.field.get_extent(margin=margin)
    
    def addField(self, new_field: PotentialField, reload_bbox=True):
        """
        Adds all RGJs from another potential field to this field and updates the quadtree accordingly.

        Args:
            new_field (PotentialField): A potential field containing RGJs to add.
            reload_bbox (bool, optional): Whether to recompute the bounding box after addition. Defaults to True.

        Returns:
            np.ndarray: Indices in the field corresponding to the newly added RGJs.
        """

        if self.quadtree.conservative:
            warnings.warn("Quadtree made not conservative")
        self.quadtree.conservative = False

        n_original = len(self.field)

        # Add RGJs to field
        
        self.field.addField(new_field=new_field, reload_bbox=reload_bbox)

        # Align new_field geometry with main field
        new_field.reload_center_point(False)
        new_field.center_point = self.field.center_point
        new_field.size = self.field.size

        # Build temporary quadtree for new_field (not conservative!)
        new_qtree = QuadTree(
            new_field,
            minimum_length_limit=self.quadtree.min_sector_size,
            maximum_length_limit=self.quadtree.max_sector_size,
            edge_bounds=self.quadtree.edge_bounds,
            size=self.quadtree.size,
            build_tree=True,
            conservative=False
        )

        # Shift all RGJ indices in new quadtree
        def update_idx(quad: QuadNode):
            if quad is None or len(quad.rgj_idx) == 0:
                return
            quad.rgj_idx += n_original
            for child in quad.children:
                update_idx(child)

        update_idx(new_qtree.root)

        # Merge new quadtree into existing one
        def update_quad(rootquad: QuadNode, newquad: Optional[QuadNode]):
            if newquad is None or newquad.boundary_zone == self.quadtree.n_zones:
                return

            # Merge zone info
            if newquad.boundary_zone < rootquad.boundary_zone:
                rootquad.boundary_zone = newquad.boundary_zone
                rootquad.boundary_max_range = newquad.boundary_max_range

            # Merge RGJ indices and zones
            if len(newquad.rgj_idx) > 0:
                rootquad.rgj_idx = np.concatenate([rootquad.rgj_idx, newquad.rgj_idx])
                rootquad.rgj_zones = np.concatenate([rootquad.rgj_zones, newquad.rgj_zones])

            # If root is leaf and new is not, convert to branch
            if rootquad.leaf and not newquad.leaf:
                self.quadtree.leaves.remove(rootquad)
                rootquad.children = [None] * len(rootquad.chdToIdx)
                rootquad.neighbors = [None] * len(rootquad.nghToIdx)
                rootquad.leaf = False

            # Recurse into children
            for child in ['tl', 'tr', 'bl', 'br']:
                nq = newquad[child]
                rq = rootquad[child]

                if rq is None:
                    self.quadtree.replace_branch(rootquad, child, nq)
                else:
                    update_quad(rq, nq)

        update_quad(self.quadtree.root, new_qtree.root)

        return np.arange(n_original, len(self.field))

    def addRGJ(self, rgj:Union[RGJDict, RGJGeometry], properties:Optional[dict] = None, reload_bbox = True, **kward) -> List[int]:

        if not isinstance(rgj, RGJGeometry):
            if not isinstance(rgj, dict) or "type" not in rgj:
                raise ValueError("RGJ must be an RGJGeometry.")
            
            cls = globals().get(rgj["type"] + "RGJ")
            if cls is None:
                raise ValueError(f"No RGJ class found for type {rgj['type']}")
            
            rgj = cls(properties=properties, **rgj, **kward)
        
        new_field = PotentialField([rgj])

        return self.addField(new_field=new_field, reload_bbox=reload_bbox)

    def delRGJ(self, idxs: Union[int, List[int]], reload_bbox=True, pop_field=False, pop_tree=False):
        """
        Deletes one or more RGJs from the potential field and updates the quadtree to reflect these deletions.

        This involves:
        - Removing the RGJs from the field.
        - Traversing and cleaning the quadtree to update indices, remove deleted references, 
        and merge empty branches if possible.
        - Optionally returning the deleted RGJs as a field or quadtree for backup or reuse.

        Args:
            idxs (int or List[int]): Index or indices of RGJs to remove.
            reload_bbox (bool): Whether to recompute the bounding box of the field. Defaults to True.
            pop_field (bool): If True, return a PotentialField containing the removed RGJs.
            pop_tree (bool): If True, return a QuadTree built on the removed RGJs.

        Returns:
            Optional[PotentialField, QuadTree, Tuple]: Depending on `pop_field` and `pop_tree`, 
                returns the removed RGJs as a PotentialField, a QuadTree, or both.
        """
        if self.quadtree.conservative:
            warnings.warn("Quadtree made non-conservative")
            self.quadtree.conservative = False

        idxs = np.atleast_1d(idxs).astype(int)
        idxs = np.unique(idxs % len(self))[::-1]  # Wrap, deduplicate, and reverse sort

        # Step 1: Store deleted RGJs before removing from field
        rgjs = [self.field.rgjs[idx] for idx in idxs]
        self.field.delRGJ(idxs, reload_bbox=reload_bbox)

        # Step 2: Build shift map (old index -> new index)
        total = len(self.field.rgjs) + len(idxs)
        shift_map = np.arange(total)
        
        deleted = np.zeros(total, dtype=bool)
        deleted[idxs] = True
        shift_map = shift_map - np.cumsum(deleted)

        def shift_recursive(quad: QuadNode):
            if quad is None or quad.rgj_idx.size == 0:
                return
            
            quad.rgj_idx = shift_map[quad.rgj_idx]
            if not quad.leaf:
                for child in ['tl', 'tr', 'bl', 'br']:
                    shift_recursive(quad[child])

        # Step 3: Traverse quadtree and update nodes
        def clean_quad(quad: QuadNode):
            if quad is None:
                return

            # --- Filter out deleted RGJ indices ---
            keep_mask = ~np.isin(quad.rgj_idx, idxs, assume_unique=True)

            # If no RGJs in this quad are being deleted, skip processing
            if np.all(keep_mask):
                shift_recursive(quad)
                return

            # --- Apply filtering ---
            quad.rgj_idx = shift_map[quad.rgj_idx[keep_mask]]
            quad.rgj_zones = quad.rgj_zones[keep_mask]

            quad.boundary_zone = (min(quad.rgj_zones) if len(quad.rgj_zones) > 0 else self.quadtree.n_zones)

            # --- Attempt early merge (before recursion) ---
            if (not quad.leaf) and quad.size <= self.quadtree.max_sector_size and quad.boundary_zone == self.quadtree.n_zones:
                self.quadtree.leaves -= self.quadtree.search_leaves(quad)
                quad.children = [None] * len(quad.chdToIdx)
                quad.neighbors = [None] * len(quad.nghToIdx)
                self.quadtree.mark_leaf(quad)
                return

            # --- Recurse if not a leaf ---
            if not quad.leaf and keep_mask.size > 0:
                for child in ['tl', 'tr', 'bl', 'br']:
                    clean_quad(quad[child])

        clean_quad(self.quadtree.root)

        # Step 4: Optionally return removed data
        if pop_field or pop_tree:
            search_field = PotentialField(rgjs)
            search_field.reload_center_point(False)
            search_field.center_point = self.field.center_point
            search_field.size = self.field.size

            if pop_field and not pop_tree:
                return search_field

            search_qtree = QuadTree(
                search_field,
                minimum_length_limit=self.quadtree.min_sector_size,
                edge_bounds=self.quadtree.edge_bounds,
                size=self.quadtree.size,
                build_tree=True
            )

            if pop_tree and not pop_field:
                return search_qtree

            return search_field, search_qtree

    def in_bbox(self, point: Point, max_depth: int = 2) -> bool:
        """
        Check if the point lies within the bounding box of any RGJ using the quadtree.

        Args:
            point (Point): The 2D point to check.
            max_depth (int): Maximum depth to search in the quadtree.

        Returns:
            bool: True if the point lies in any RGJ bounding box.
        """
        point = np.array(point, dtype=np.float64)
        quad_chain = self.quadtree.find_quads_chain([point], max_depth=max_depth)[0]

        searched_rgj = set()

        for quad in reversed(quad_chain):
            to_search_rgj = set(quad.rgj_idx) - searched_rgj

            if len(to_search_rgj):
                searched_rgj.update(to_search_rgj)

                bbox_idx = self.field.find_bbox(point, filted_idx=to_search_rgj)

                if len(bbox_idx):
                    return bbox_idx

        # fallback: search entire field if no relevant RGJs found
        return self.field.in_bbox(point)
    
    def find_bbox(self, point: Point, max_depth: int = 2) -> np.ndarray:
        """
        Return indices of RGJs whose bounding boxes contain the point, using the quadtree.

        Args:
            point (Point): The 2D point to check.
            max_depth (int): Maximum depth of the quadtree to consider.

        Returns:
            np.ndarray: Indices of RGJs (global) whose bounding boxes contain the point.

        Note function may return only a subset of the RGJs indexes (those closest in distance)
        """
        point = np.array(point, dtype=np.float64)
        quad_chain = self.quadtree.find_quads_chain([point], max_depth=max_depth)[0]

        searched_rgj = set()

        for quad in reversed(quad_chain):
            to_search_rgj = set(quad.rgj_idx) - searched_rgj

            if len(quad.rgj_idx):
                searched_rgj.update(to_search_rgj)

                bbox_idx = self.field.find_bbox(point, filted_idx=to_search_rgj)

                if len(bbox_idx):
                    return bbox_idx

        # fallback: search whole field if no relevant RGJs found; not expected to ever happen
        return self.field.find_bbox(point)
    
    def repulsion_vectors(
        self,
        points: Union[np.ndarray, List['Point']],
        min_dist_select: bool = True,
        return_reference: bool = False,
        max_depth: int = 2
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Computes repulsion vectors for 2D points using only relevant RGJs from a quadtree field.
        Each point is assigned to the deepest quad with non-empty RGJs, and repulsion vectors
        are calculated using those RGJ indices. Redundant calculations are avoided by grouping
        points per quad.

        Args:
            points (Union[np.ndarray, List[Point]]): Input 2D points.
            min_dist_select (bool): Whether to use minimum distance RGJ filtering.
            return_reference (bool): If True, returns RGJ indices used per point.
            max_depth (int): Maximum depth in the quadtree to traverse.

        Returns:
            If return_reference is False:
                np.ndarray of shape (N, 2): Repulsion vectors for all input points.
                
            If return_reference is True:
                Tuple[np.ndarray, np.ndarray]: (repulsion_vectors, rgj_indices_used_per_point)
        """
        points = np.atleast_2d(points).astype(float)
        n_points = len(points)
        repulsion_vectors = np.zeros((n_points, 2), dtype=float)
        rgj_reference_ids = np.zeros(n_points, dtype=int) if return_reference else None

        unique_quads, quad_to_point_indices = self.__group_points_by_quads_with_rgjs(points, max_depth=max_depth)

        for quad_idx, pt_indices in quad_to_point_indices.items():
            quad = unique_quads[quad_idx]
            rgj_indices = quad.rgj_idx

            group_points = points[pt_indices]
            if return_reference:
                group_vectors, group_rgj_ids = self.field.repulsion_vectors(
                    points=group_points,
                    filted_idx=rgj_indices,
                    min_dist_select=min_dist_select,
                    return_reference=True
                )
                repulsion_vectors[pt_indices] = group_vectors
                rgj_reference_ids[pt_indices] = group_rgj_ids
            else:
                group_vectors = self.field.repulsion_vectors(
                    points=group_points,
                    filted_idx=rgj_indices,
                    min_dist_select=min_dist_select,
                    return_reference=False
                )
                repulsion_vectors[pt_indices] = group_vectors

        if return_reference:
            return repulsion_vectors, rgj_reference_ids
        else:
            return repulsion_vectors
        
    def gradient(self, points, min_dist_select=True, max_depth=2):
        points = np.atleast_2d(points).astype(float)
        n_points = len(points)

        if len(self.field) == 0:
            return np.zeros((n_points, 2), dtype=float)

        grad = np.zeros((n_points, 2), dtype=float)

        unique_quads, quad_to_point_indices = self.__group_points_by_quads_with_rgjs(points, max_depth=max_depth)

        for quad_idx, pt_indices in quad_to_point_indices.items():
            quad = unique_quads[quad_idx]
            points_in_quad = points[pt_indices]

            for rgj_idx in quad.rgj_idx:
                grad_vecs = self.field.rgjs[rgj_idx].gradient(points_in_quad, min_dist_select=min_dist_select)
                grad[pt_indices] += grad_vecs

        return grad

    def eval(self, points: Union[np.ndarray, List['Point']], max_depth=2) -> np.ndarray:
        """
        Evaluate the potential field at given points using quadtree
        to filter relevant RGJs efficiently.

        Args:
            points (Union[np.ndarray, List[Point]]): Points to evaluate (Nx2).

        Returns:
            np.ndarray: Evaluated potential values at each point.
        """
        points = np.atleast_2d(points).astype(float)
        n_points = len(points)

        if not len(self.field.rgjs):
            return np.zeros(n_points, dtype=float)

        unique_quads, quad_to_point_indices = self.__group_points_by_quads_with_rgjs(points, max_depth=max_depth)

        results = np.zeros(n_points, dtype=float)

        for quad_idx, pt_indices in quad_to_point_indices.items():
            quad = unique_quads[quad_idx]
            rgj_indices = quad.rgj_idx
            group_points = points[pt_indices]

            if not len(rgj_indices):
                results[pt_indices] = 0.0
            else:
                results[pt_indices] = self.field.eval(group_points, filted_idx=rgj_indices)

        return results
    
    def eval_per(self, points: Union[np.ndarray, List[Point]], idxs:Optional[List[int]] = None) -> np.ndarray:
        return self.field.eval_per(points=points, idxs=idxs)
    
    def squared_dist(self, points:Union[np.ndarray, List[Point]], scaled=True, inverted=True, max_depth=2, return_reference = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        points = np.array(points)
        if not len(self):
            warnings.warn("There are not any RGJs elements in the field")
            if return_reference:
                return points.sum(1)*np.inf, -np.ones_like(points.sum(1))
            return points.sum(1)*np.inf

        dists = self.squared_dist_list(points=points, scaled=scaled, inverted=inverted, max_depth=max_depth)
        if return_reference:
            min_idxs = np.argmin(dists, axis=1)
            return dists[np.arange(len(dists)), min_idxs], min_idxs

        return np.min(dists, axis=1)
    
    def squared_dist_per(self, points: Union[np.ndarray, List[Point]], idxs:Optional[List[int]] = None, scaled=True, inverted=True) -> np.ndarray:
        return self.field.squared_dist_per(points=points, idxs=idxs, scaled=scaled, inverted=inverted)
    
    def squared_dist_list(
        self,
        points: Union[np.ndarray, List['Point']],
        scaled: bool = True,
        inverted: bool = True,
        max_depth: int = 2
    ) -> np.ndarray:
        """
        Compute squared distances from each point to all RGJs in the field, but only
        evaluate the ones relevant per point using the quadtree. Non-relevant RGJs will
        be set to np.inf to maintain consistent output shape.

        Args:
            points (Union[np.ndarray, List[Point]]): Input points, shape (N, 2).
            scaled (bool): Whether to scale distances.
            inverted (bool): Whether to invert distance values.
            max_depth (int): Max depth of quadtree search.

        Returns:
            np.ndarray: Squared distances of shape (N, M), where M = total RGJs.
        """
        points = np.atleast_2d(points).astype(float)
        n_points = len(points)
        total_rgjs = len(self.field.rgjs)

        if total_rgjs == 0:
            warnings.warn("There are no RGJs in the field.")
            return np.ones((n_points, 1)) * np.inf

        dist_matrix = np.ones((n_points, total_rgjs), dtype=np.float64) * np.inf

        # Group by quads that contain RGJs
        unique_quads, quad_to_point_indices = self.__group_points_by_quads_with_rgjs(points, max_depth=max_depth)

        for quad_idx, pt_indices in quad_to_point_indices.items():
            quad = unique_quads[quad_idx]
            rgj_indices = quad.rgj_idx
            if not rgj_indices:
                continue

            group_points = points[pt_indices]
            group_dists = np.stack([
                self.field.rgjs[i].squared_dist(group_points, scaled=scaled, inverted=inverted)
                for i in rgj_indices
            ], axis=1)

            for i, pt_i in enumerate(pt_indices):
                for j, rgj_i in enumerate(rgj_indices):
                    dist_matrix[pt_i, rgj_i] = group_dists[i, j]

        return dist_matrix

    def estimate_route_area(self, route:Union[List[Point], np.ndarray], step=1e-3, n=0, scale_transform:Scaler = lambda x: x, max_depth:int = 2) -> float:
        route = np.array(route)

        points, step, _ = lpf.interpolate_along_route(route=route, step=step, n=n, return_step_n=True)
        points = points if n <= 0 else points[:-1]

        f_eval = scale_transform(self.eval(points=points, max_depth=max_depth))

        return f_eval.sum()*step
    
    def estimate_route_highest_potential(self, route:Union[List[Point], np.ndarray], step=1e-2, n=0, scale_transform:Scaler = lambda x: x, max_depth:int = 2) -> float:
        route = np.array(route)

        points, step, _ = lpf.interpolate_along_route(route=route, step=step, n=n, return_step_n=True)
        points = points if n <= 0 else points[:-1]

        f_eval:np.ndarray = scale_transform(self.eval(points=points, max_depth=max_depth))

        return f_eval.max()

    def to_image(self, resolution:int = 400, margin:float = 0.0, center_point:Optional[Point] = None, size:Optional[FieldSize] = None, max_depth:int = 2, return_extent=True) -> np.ndarray:

        if center_point is None:
            if self.field.center_point is None:
                raise RuntimeError('center point for field has not been defined')
            
            center_point = self.field.center_point

        if size is None:
            if self.field.size is None:
                raise RuntimeError('size of field has not been defined')

            size = self.field.size
        else:
            size = np.array(size)

        n2 = size/2.0

        loc_tl = np.array(center_point) + np.array([-n2[0]-margin, n2[1]+margin])
        loc_br = np.array(center_point) + np.array([n2[0]+margin, -n2[1]-margin])

        y_resolution = int(resolution*abs(loc_tl[1] - loc_br[1])/abs(loc_br[0] - loc_tl[0]))
        xaxis = np.linspace(loc_tl[0], loc_br[0], resolution)
        yaxis = np.linspace(loc_tl[1], loc_br[1], y_resolution)

        xgrid, ygrid = np.meshgrid(xaxis, yaxis)
        points = np.vstack([xgrid.ravel(), ygrid.ravel()]).T

        image = self.eval(points, max_depth=max_depth).reshape((y_resolution, resolution))

        if return_extent:
            extent = np.reshape([loc_tl[0], loc_br[0], loc_br[1], loc_tl[1]], -1).tolist()
            return image, extent

        return image
    

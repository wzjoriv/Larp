from collections import defaultdict
import heapq
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from larp.field import PotentialField
from larp.pp.network import QuadNetwork

from larp.quad import QuadNode, QuadTree
from larp.types import Scaler, Point

"""
Author: Josue N Rivera

Module providing path planning algorithms. They can be over a potentiald field or quadtree-based spatial network of the potential field.
Implements classic and multi-resolution planning strategies including A* and Dijkstra.
"""

PathAlg = Callable[[Point, Point, QuadNetwork, dict, Any], Optional[List[Point]]]
PathAlgArg = Union[str, PathAlg]

def __reconstruct_quad_path__(came_from:dict, current:QuadNode):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def __get_scaled_distance__(entry:Point, exit:Point, node_to:QuadNode, scaler:Optional[Scaler]=None, max_scale: float = np.inf):

        if scaler is None:
            multipler = 1.0
        else:
            multipler = np.inf if node_to.boundary_zone == 0 else scaler(node_to.boundary_max_range)
        
        multipler = min(multipler, max_scale)

        diff = exit - entry

        return multipler*np.linalg.norm(diff)

def optimize_path_via_edge_bundling(path: List[Point], quad_path: List[QuadNode], network: QuadNetwork) -> List[Point]:

    # TODO: Fix 
    def quads_aligned(q1, q2):
        """Check if two quads are adjacent and aligned either horizontally or vertically (not diagonally)."""
        if q1 is None or q2 is None:
            return False

        shared_edge = q1.get_shared_edge(q2)
        if shared_edge is None:
            return False

        # Not a single-point overlap (i.e., not diagonal)
        return not np.allclose(shared_edge[0], shared_edge[1])

    if not path is not None or not quad_path is not None or len(path) != len(quad_path) + 1:
        raise ValueError("Path and quad_path length mismatch. Expected len(path) == len(quad_path) + 1.")

    optimized_points = [path[0]]
    i = 0

    while i < len(quad_path):
        # Find longest aligned segment in quad_path
        j = i + 1
        while j < len(quad_path) and quads_aligned(quad_path[j - 1], quad_path[j]):
            j += 1

        # Segment start and end
        start_point = optimized_points[-1]
        end_point = path[j] if j < len(path) else path[-1]

        line_vec = end_point - start_point
        num_steps = j - i

        for k in range(1, num_steps):
            t = k / num_steps
            interp = start_point + t * line_vec
            current_quad = quad_path[i + k - 1]
            next_quad = quad_path[i + k]

            shared = network.get_shared_entry_point(current_quad, next_quad, interp)
            optimized_points.append(shared if shared is not None else interp)

        optimized_points.append(end_point)
        i = j  # Advance to the next segment

    return optimized_points

def has_quad_zone_sight(
        p1: np.ndarray,
        p2: np.ndarray,
        network: QuadNetwork,
        step: Optional[float] = None,
        min_zone:int = 1
    ) -> Tuple[bool, Optional[int], Optional[QuadNode]]:

    """
    Returns whether the path from p1 to p2 maintains or increases zone (i.e., risk doesn't increase).

    Args:
        p1 (np.ndarray): Start point.
        p2 (np.ndarray): End point.
        network (QuadNetwork): The quad network to check in.
        step (float): Distance between sample points.

    Returns:
        bool: True if zone doesn't decrease, False otherwise.
    """
    if step is None:
        step = network.quadtree.min_sector_size / 4

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    if np.allclose(p1, p2):
        return True

    distance = np.linalg.norm(p2 - p1)
    num_samples = max(2, int(distance / step))
    alphas = np.linspace(0, 1, num_samples)
    points = (1 - alphas[:, None]) * p1 + alphas[:, None] * p2

    quads = network.find_quad(points)
    allowed_zone = max(quads[0].boundary_zone, min_zone)

    # Reduce to unique quads
    used = set()
    quads = [x for x in quads if x not in used and (used.add(x) or True)]

    # Accept quads that have a non-decending quad zone order 
    for quad in quads:
        if quad is None or quad.boundary_zone < allowed_zone:
            return False
        
        allowed_zone = quad.boundary_zone

    return True

def network_path_smoothing(path: List[Point], network: QuadNetwork, step: Optional[float] = None) -> np.ndarray:
    """
    Smooths a path by merging segments that are in the same zone or higher zone.
    """
    if len(path) <= 2:
        return np.array(path, dtype=float)

    optimized = [path[0]]

    i = 1
    while i < len(path) - 1:
        if not has_quad_zone_sight(optimized[-1], path[i+1], network=network, step=step):
            optimized.append(path[i])
        i += 1

    optimized.append(path[-1])
    return np.array(optimized, dtype=float)

class Planner():

    """
    Course path without dynamics included
    """

    def __init__(self, alg: Optional[PathAlgArg] = None):

        self.algs = defaultdict(lambda: None)
        self.alg = self.algs['custom']
        self.memory = {}

        if alg is None:
            self.select_alg(alg=alg, reset_memory=False)

    def refresh(self):
        """
        Resets planner memory.
        """
        self.reset_memory()

    def reset_memory(self):
        """
        Clears any memory state stored across planning runs.
        """
        self.memory.clear()

    def add_alg(self, name: str, algorithm: PathAlg):
        """
        Adds a custom path planning algorithm.

        Args:
            name (str): Name identifier for the custom algorithm.
            algorithm (PathAlg): The algorithm function.
        """

        self.algs[name.lower()] = algorithm

    def select_alg(self, alg: PathAlgArg, reset_memory = True):
        """
        Selects a planning algorithm by name or function reference.

        Args:
            alg (Optional[PathAlgArg]): Algorithm name or custom function.
            reset_memory (bool): Whether to reset internal memory state.
        """

        if isinstance(alg, str):
            if alg not in self.algs.keys():
                raise KeyError("Selected algorithm is not available")

            self.alg = self.algs[alg.lower()]
        else:
            self.add_alg('custom', alg)
            self.alg = self.algs["custom"]

        if reset_memory:
            self.reset_memory()

    def find_path(self, start_point:Point, end_point:Point, reset_memory = False, **kargs) -> Union[List[Point], np.ndarray]:
        """
        Executes the selected path planning algorithm from start to goal.

        Args:
            start_point (Point): Starting position.
            end_point (Point): Target position.
            reset_memory (bool): Whether to reset memory before planning.

        Returns:
            Optional[Union[List[Point], np.ndarray]]: Path as a list of points, or None if no path found.
        """

        raise NotImplementedError

class FieldPlanner(Planner):

    """
    Path planner based on field only
    """

    def __init__(self, field:Union[PotentialField], alg: Optional[PathAlgArg] = None):
        
        self.field = field

        self.algs = defaultdict(lambda: None)

        self.alg = None
        self.memory = {}
        self.select_alg(alg=alg, reset_memory=False)

class QuadPlanner(Planner):

    """
    Path planner using quad network
    """

    def __init__(self, quadtree:QuadTree, alg: Optional[PathAlgArg] = None):

        Planner.__init__(self, None)
        
        self.field = quadtree.field
        self.quadtree = quadtree
        self.network = QuadNetwork(quadtree=quadtree)

        self.algs["a*"] = find_path_A_star
        self.algs["dijkstra"] = find_path_dijkstra
        self.algs["a*-e"] = find_path_astar_e
        self.algs["dijkstra-e"] = find_path_dijkstra_e
        self.alg = self.algs["a*"]

        if alg is not None:
            self.select_alg(alg=alg, reset_memory=False)

    def select_alg(self, alg, reset_memory=True):
        """
        Selects a planning algorithm by name or function reference.

        Args:
            alg (Optional[PathAlgArg]): Algorithm name or custom function.
            reset_memory (bool): Whether to reset internal memory state.

        Options:
        - a*
        - dijkstra
        - a*-e
        - dijkstra-e
        """

        return super().select_alg(alg, reset_memory)

    def refresh(self):
        """
        Resets planner memory and refreshes the network graph.
        """
        super().refresh()
        self.refresh_network()

    def refresh_network(self):
        """
        Refreshes the quad connectivity in the network graph.
        """
        self.network.refresh()

    def find_path(self, start_point:Point, end_point:Point, refresh_network = True, reset_memory = False, smooth_path = True, **kargs) -> Optional[Union[List[Point], np.ndarray]]:
        """
        Executes the selected path planning algorithm from start to goal.

        Args:
            start_point (Point): Starting position.
            end_point (Point): Target position.
            refresh_network (bool): Whether to refresh quad connectivity.
            reset_memory (bool): Whether to reset memory before planning.

        Returns:
            Optional[Union[List[Point], np.ndarray]]: Path as a list of points, or None if no path found.
        """
        if refresh_network:
            self.refresh_network()

        if reset_memory:
            self.memory.clear()

        start_point, end_point = np.asarray(start_point, dtype=float), np.asarray(end_point, dtype=float)

        path = self.alg(start_point=start_point, end_point=end_point, network=self.network, memory=self.memory, **kargs)

        if smooth_path and path is not None:
            path = network_path_smoothing(path=path, network=self.network)

        return  path


# Path planning algorithms

def find_path_A_star(start_point:Point, end_point:Point, network:QuadNetwork, scaler:Optional[Scaler]=None, max_scale: float = np.inf, **kargs) -> Optional[List[Point]]:
    """
    A* pathfinding algorithm over the quad network using quad centers as waypoints.

    Args:
        start_point (Point): Starting position.
        end_point (Point): Goal position.
        network (QuadNetwork): Graph of connected QuadNodes.
        scaler (Optional[Scaler]): Optional scaling function for traversal cost.
        max_scale (float): Upper limit for scaled traversal cost.

    Returns:
        Optional[List[Point]]: Path from start to goal, or None if no path exists.
    """
    start_quad, end_quad = tuple(network.find_quad([start_point, end_point]))

    if scaler is None:
        scaler = lambda p: 1.0 + 2*p

    open_set = []
    heapq.heappush(open_set, (0, start_quad))

    came_from = {}
    g_score = defaultdict(lambda: np.inf)
    g_score[start_quad] = 0

    f_score = defaultdict(lambda: np.inf)
    f_score[start_quad] = network.get_center_distance(start_quad, end_quad)

    current = None

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end_quad:
            break

        for neighbor in network[current]:
            tentative_g_score = g_score[current] + network.get_center_distance(current, neighbor, scaler=scaler, max_scale=max_scale)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + network.get_center_distance(neighbor, end_quad)

                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    if current == end_quad:
        quad_path = __reconstruct_quad_path__(came_from, current)
        path = np.array([start_point] + network.get_quad_center(quad_path)[1:-1] + [end_point])

        path = [start_point] + [network.get_shared_entry_point(quad_path[i], quad_path[i+1], path[i+1]) for i in range(len(quad_path)-1)] + [end_point]

        path = optimize_path_via_edge_bundling(path, quad_path, network)

        return np.array(path)

    return None

def find_path_dijkstra(start_point:Point, end_point:Point, network:QuadNetwork, scaler:Optional[Scaler]=None, max_scale: float = np.inf, **kargs)-> Optional[List[Point]]:
    """
    Dijkstra's algorithm over the quad network, using quad centers as waypoints.

    Args:
        start_point (Point): Starting position.
        end_point (Point): Goal position.
        network (QuadNetwork): Graph of connected QuadNodes.
        scaler (Optional[Scaler]): Optional scaling function for traversal cost.
        max_scale (float): Upper limit for scaled traversal cost.

    Returns:
        Optional[List[Point]]: Path from start to goal, or None if no path exists.
    """
    start_quad, end_quad = network.find_quad([start_point, end_point])

    if scaler is None:
        scaler = lambda p: 1.0 + p

    open_set = []
    heapq.heappush(open_set, (0, start_quad))

    came_from = {}
    dist = defaultdict(lambda: np.inf)
    dist[start_quad] = 0
    current = None

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end_quad:
            break

        for neighbor in network[current]:
            tentative_dist = dist[current] + network.get_center_distance(current, neighbor, scaler=scaler, max_scale=max_scale)

            if tentative_dist < dist[neighbor]:
                came_from[neighbor] = current
                dist[neighbor] = tentative_dist

                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (dist[neighbor], neighbor))

    if current == end_quad:
        quad_path = __reconstruct_quad_path__(came_from, current)
        path = np.array([start_point] + network.get_quad_center(quad_path)[1:-1] + [end_point])

        path = [start_point] + [network.get_shared_entry_point(quad_path[i], quad_path[i+1], path[i+1]) for i in range(len(quad_path)-1)] + [end_point]
        path = optimize_path_via_edge_bundling(path, quad_path, network)

        return np.array(path)

    return None

def find_path_astar_e(
        start_point: Point,
        end_point: Point,
        network: QuadNetwork,
        scaler: Optional[Scaler] = None,
        max_scale: float = np.inf,
        **kargs
    ) -> Optional[List[Point]]:

    """
    A* pathfinding with edge-aware traversal between quad regions.

    This variant of A* seeks to avoid quad center traversal and instead moves between
    shared edges based on entry direction and local quad structure.

    Args:
        start_point (Point): Starting position.
        end_point (Point): Target position.
        network (QuadNetwork): The graph of quad connectivity.
        scaler (Optional[Scaler]): Optional scaling function for local quad cost.
        max_scale (float): Maximum value allowed by the scaler.

    Returns:
        Optional[List[Point]]: Point-to-point path, or None if no path is found.
    """

    start_point = np.array(start_point, dtype=float)
    end_point = np.array(end_point, dtype=float)

    start_quad, end_quad = network.find_quad([start_point, end_point])

    if start_quad is None or end_quad is None:
        return None

    if np.allclose(start_point, end_point):
        return [start_point]
    
    if start_quad == end_quad:
        return [start_point, end_point]

    if scaler is None:
        scaler = lambda p: 1.0 + p

    open_set = []
    open_set_hash = set()
    heapq.heappush(open_set, (0, start_quad))
    open_set_hash.add(start_quad)

    came_from = {}
    g_score = defaultdict(lambda: np.inf)
    g_score[start_quad] = 0

    f_score = defaultdict(lambda: np.inf)
    f_score[start_quad] = np.linalg.norm(start_point - end_point)

    edge_entry = {start_quad: start_point}

    current = None

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        if current == end_quad:
            break

        for neighbor in network[current]:
            entry = edge_entry[current]
            exit = network.get_shared_entry_point(current, neighbor, entry)

            if exit is None:
                continue

            tentative_g = g_score[current] + __get_scaled_distance__(entry, exit, neighbor, scaler, max_scale)

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + np.linalg.norm(exit - end_point)
                edge_entry[neighbor] = exit

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    if current == end_quad:
        # Reconstruct full point-level and quad path
        path = [end_point]
        quad = end_quad
        while quad in came_from:
            path.append(edge_entry[quad])
            quad = came_from[quad]

        path.append(start_point)
        path = path[::-1]

        quad_path = __reconstruct_quad_path__(came_from, end_quad)
        path = optimize_path_via_edge_bundling(path, quad_path, network)

        return np.array(path)

    return None

def find_path_dijkstra_e(
        start_point: Point,
        end_point: Point,
        network: QuadNetwork,
        scaler: Optional[Scaler] = None,
        max_scale: float = np.inf,
        **kargs
    ) -> Optional[List[Point]]:

    """
    Dijkstra's algorithm with edge-aware traversal between quad regions.

    This variant of Dijkstra seeks to avoid quad center traversal and instead moves between
    shared edges based on entry direction and local quad structure.

    Args:
        start_point (Point): Starting position.
        end_point (Point): Target position.
        network (QuadNetwork): The graph of quad connectivity.
        scaler (Optional[Scaler]): Optional scaling function for local quad cost.
        max_scale (float): Maximum value allowed by the scaler.

    Returns:
        Optional[List[Point]]: Point-to-point path, or None if no path is found.
    """

    start_point = np.array(start_point, dtype=float)
    end_point = np.array(end_point, dtype=float)

    start_quad, end_quad = network.find_quad([start_point, end_point])

    if start_quad is None or end_quad is None:
        return None

    if np.allclose(start_point, end_point):
        return [start_point]
    
    if start_quad == end_quad:
        return [start_point, end_point]

    if scaler is None:
        scaler = lambda p: 1.0 + p

    open_set = []
    open_set_hash = set()
    heapq.heappush(open_set, (0, start_quad))
    open_set_hash.add(start_quad)

    came_from = {}
    g_score = defaultdict(lambda: np.inf)
    g_score[start_quad] = 0

    edge_entry = {start_quad: start_point}

    current = None

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        if current == end_quad:
            break

        for neighbor in network[current]:
            entry = edge_entry[current]
            exit = network.get_shared_entry_point(current, neighbor, entry)

            if exit is None:
                continue

            tentative_g = g_score[current] + __get_scaled_distance__(entry, exit, neighbor, scaler, max_scale)

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                edge_entry[neighbor] = exit

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (g_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    if current == end_quad:
        # Reconstruct full point-level path
        path = [end_point]
        quad = end_quad
        while quad in came_from:
            path.append(edge_entry[quad])
            quad = came_from[quad]
        path.append(start_point)
        path = path[::-1]

        quad_path = __reconstruct_quad_path__(came_from, end_quad)
        path = optimize_path_via_edge_bundling(path, quad_path, network)

        return np.array(path)

    return None

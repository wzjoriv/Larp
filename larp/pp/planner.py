from collections import defaultdict
import heapq
from typing import Any, Callable, List, Optional, Union

import numpy as np
from larp.field import PotentialField
from larp.pp.network import QuadNetwork

from larp.quad import QuadNode, QuadTree
from larp.types import Scaler, Point

"""
Author: Josue N Rivera

Module providing path planning algorithms over a quadtree-based spatial network.
Implements classic and multi-resolution planning strategies including A*, Dijkstra,
and Multi-resolution Field D* (MRF-D*).
"""

PathAlg = Callable[[Point, Point, QuadNetwork, dict, Any], Optional[List[Point]]]
PathAlgArg = Union[str, PathAlg]

def __reconstruct_quad_path__(came_from:dict, current:QuadNode):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]


class Planner():

    """
    Course path without dynamics included
    """

    def __init__(self, alg: Optional[PathAlgArg]):

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
            self.alg = self.algs[alg]
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

    def __init__(self, field:PotentialField, alg: Optional[PathAlgArg]):
        
        self.field = field

        self.algs = defaultdict(lambda: None)

        self.alg = None
        self.memory = {}
        self.select_alg(alg=alg, reset_memory=False)

class QuadPlanner(Planner):

    """
    Path planner using quad network
    """

    def __init__(self, quadtree:QuadTree, alg: Optional[PathAlgArg]):

        Planner.__init__(self, None)
        
        self.field = quadtree.field
        self.quadtree = quadtree
        self.network = QuadNetwork(quadtree=quadtree)

        self.algs["a*"] = find_path_A_star
        self.algs["dijkstra"] = find_path_dijkstra
        self.algs["mrfd*"] = find_path_mrfdstar
        self.alg = self.algs["a*"]

        if alg is not None:
            self.select_alg(alg=alg, reset_memory=False)

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

    def find_path(self, start_point:Point, end_point:Point, refresh_network = False, reset_memory = False, **kargs) -> Optional[Union[List[Point], np.ndarray]]:
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
        if reset_memory:
            self.memory.clear()
        if refresh_network:
            self.refresh_network()

        return self.alg(start_point=start_point, end_point=end_point, network=self.network, memory=self.memory, **kargs)


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
        scaler = lambda p: 1.0 + p

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
        return np.array([start_point] + network.get_quad_center(quad_path)[1:-1] + [end_point])

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
    start_quad, end_quad = tuple(network.find_quad([start_point, end_point]))

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
        return np.array([start_point] + network.get_quad_center(quad_path)[1:-1] + [end_point])

    return None

def find_path_mrfdstar(
        start_point: Point,
        end_point: Point,
        network: QuadNetwork,
        scaler: Optional[Scaler] = None,
        max_scale: float = np.inf,
        **kargs
    ) -> Optional[List[Point]]:

    """
    Multi-resolution Field D* (MRF-D*) pathfinding with edge-aware traversal between quad regions.

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
    
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    start_quad, end_quad = tuple(network.find_quad([start_point, end_point]))

    if scaler is None:
        scaler = lambda p: 1.0 + p

    open_set = []
    heapq.heappush(open_set, (0, start_quad))

    came_from = {}
    g_score = defaultdict(lambda: np.inf)
    g_score[start_quad] = 0

    f_score = defaultdict(lambda: np.inf)
    f_score[start_quad] = np.linalg.norm(start_point - end_point)

    edge_entry = {start_quad: start_point}

    current = None

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end_quad:
            break

        for neighbor in network[current]:
            entry = edge_entry[current]
            exit = current.get_shared_entry_point(neighbor, entry)

            if exit is None:
                continue

            traversal_cost = np.linalg.norm(exit - entry)
            scaled_cost = traversal_cost * min(scaler(neighbor.boundary_max_range), max_scale)

            tentative_g = g_score[current] + scaled_cost

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + np.linalg.norm(exit - end_point)
                edge_entry[neighbor] = exit
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    if current == end_quad:
        # Reconstruct full point-level path
        path = []
        quad = end_quad
        while quad in came_from:
            path.append(edge_entry[quad])
            quad = came_from[quad]
        path.append(start_point)
        return np.array(path[::-1] + [end_point])

    return None
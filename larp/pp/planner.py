from collections import defaultdict
import heapq
from typing import Any, Callable, List, Optional, Union

import numpy as np
from larp.pp.network import QuadNetwork

from larp.quad import QuadNode, QuadTree
from larp.types import Scaler, Point

"""
Author: Josue N Rivera
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

    def __init__(self, quadtree:QuadTree, alg: Optional[PathAlgArg]):
        
        self.field = quadtree.field
        self.quadtree = quadtree
        self.network = QuadNetwork(quadtree=quadtree)

        self.algs = defaultdict(lambda: find_path_A_star)
        self.algs["a*"] = find_path_A_star
        self.algs["dijkstra"] = find_path_dijkstra

        self.alg = None
        self.memory = {}
        self.select_alg(alg=alg, reset_memory=False)

    def refresh(self):
        self.reset_memory()
        self.refresh_network()

    def refresh_network(self):
        self.network.refresh()

    def reset_memory(self):
        self.memory.clear()

    def select_alg(self, alg: Optional[PathAlgArg] = None, reset_memory = True):

        if isinstance(alg, str):
            self.alg = self.algs[alg]
        elif alg is None:
            self.alg = self.algs["a*"]
        else:
            self.add_alg('custom', alg)
            self.alg = self.algs["custom"]

        if reset_memory:
            self.reset_memory()

    def add_alg(self, name: str, algorithm: PathAlg):
        """
        Adds a custom path planning algorithm to the network.

        Args:
            name (str): Name to reference the algorithm by.
            algorithm (PathAlg): The path planning algorithm function to use for pathfinding.
        """

        self.algs[name.lower()] = algorithm

    def find_path(self, start_point:Point, end_point:Point, reset_memory = False, **kargs) -> Union[List[Point], np.ndarray]:
        """
        Options:
            - A*
            - Dijkstra
            - [Any algorithm included by user]
        """
        if reset_memory:
            self.memory.clear()

        return self.alg(start_point=start_point, end_point=end_point, network=self.network, memory=self.memory, **kargs)


def find_path_A_star(start_point:Point, end_point:Point, network:QuadNetwork, scaler:Optional[Scaler]=None, max_scale: float = np.inf, **kargs) -> Optional[List[Point]]:
    """ find path (A*) 

    Returns None if no path found
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

def find_path_dijkstra(start_point:Point, end_point:Point, network:QuadNetwork, scaler:Optional[Scaler]=None, max_scale: float = np.inf)-> Optional[List[Point]]:
    """ find path (Dijkstra) 

    Returns None if no path found
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
from collections import defaultdict
import heapq
from typing import Callable, List, Optional, Tuple

import numpy as np

from larp.quad import QuadNode, QuadTree
from larp.types import FieldScaleTransform, Point, RoutingAlgorithmStr

"""
Author: Josue N Rivera
"""

RoutingAlgorithm = Callable[[QuadNode, QuadNode, FieldScaleTransform, dict], Optional[List[QuadNode]]]

class Network(object):
    """ Network data structure, undirected by default. 
    
    - Adapted from: https://stackoverflow.com/questions/19472530/representing-graphs-data-structure-in-python
    """

    def __init__(self, connections = None, directed = False):
        self._graph = defaultdict(set)
        self._directed = directed
        if connections is not None:
            self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to network """

        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def add_one_to_many(self, node1, nodes, overwrite_directed=False):
        """ Add connection between node1 and all other nodes """

        self._graph[node1].update(nodes)
        if not self._directed and not overwrite_directed:
            for node2 in nodes:
                self._graph[node2].add(node1)

    def remove(self, node):
        """ Remove all references to node """

        for _, cxns in self._graph.items():  # python3: items(); python2: iteritems()
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """

        return node1 in self._graph and node2 in self._graph[node1]

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
    
    def find_path(self, node1, node2):
        raise NotImplementedError
    
class RoutingNetwork(Network):

    ChildNeighOuterEdges = { # Maps child quad' outer edge to neighbors' children
        'tl': {
            't':  [('t', 'bl'), ('tr', 'br')],
            'tl': [('tl', 'br')],
            'l':  [('l', 'tr'), ('bl', 'br')]
        },
        'tr': {
            't':  [('t', 'br'), ('tl', 'bl')],
            'tr': [('tr', 'bl')],
            'r':  [('r', 'tl'), ('br', 'bl')]
        },
        'bl': {
            'b':  [('b', 'tl'), ('br', 'tr')],
            'bl': [('bl', 'tr')],
            'l':  [('l', 'br'), ('tl', 'tr')]
        },
        'br': {
            'b':  [('b', 'tr'), ('bl', 'tl')],
            'br': [('br', 'tl')],
            'r':  [('r', 'bl'), ('tr', 'tl')]
        }  
    }

    NeighOuterEdges = { # Maps quad' outer edge to neighbors' children
        'tl': ['br'],
        't':  ['bl', 'br'],
        'tr': ['bl'],
        'r':  ['tl', 'bl'],
        'br': ['tl'],
        'b':  ['tl', 'tr'],
        'bl': ['tr'],
        'l':  ['tr', 'br']
    }

    def __init__(self, quadtree:QuadTree, directed:bool=False, build_network:bool=False):
        self.quadtree = quadtree
        super().__init__(directed=directed)

        self.routing_algs = defaultdict(lambda: self.find_path_A_star)
        self.routing_algs["a*"] = self.find_path_A_star
        self.routing_algs["dijkstra"] = self.find_path_dijkstra

        if build_network:
            self.build()

    def add_routing_algorithm(self, name:str, algorithm:RoutingAlgorithm):
        self.routing_algs[name.lower()] = algorithm

    def __fill_shallow_neighs__(self, root:Optional[QuadNode] = None):

        def outer_edge_fill(quad:QuadNode, child = 'tl', side = 't'):
            child_quad = quad[child]
            parent_neigh:QuadNode = quad[[side]][0]

            subs = self.ChildNeighOuterEdges[child][side]
            if parent_neigh is None or parent_neigh.leaf:
                child_quad[[child_neigh for child_neigh, _ in subs]] = parent_neigh
            else:
                for child_neigh, neigh_child in subs:
                    child_quad[[child_neigh]] = parent_neigh[neigh_child]

        def dfs(quad:QuadNode):
            if quad.leaf: return

            qtl, qtr, qbl, qbr = quad['tl'], quad['tr'], quad['bl'], quad['br']

            # tl neighbors
            qtl[['r']] = qtr
            qtl[['br']] = qbr
            qtl[['b']] = qbl
            outer_edge_fill(quad, 'tl', 't')
            outer_edge_fill(quad, 'tl', 'tl')
            outer_edge_fill(quad, 'tl', 'l')
            # tr neighbors
            qtr[['l']] = qtl
            qtr[['bl']] = qbl
            qtr[['b']] = qbr
            outer_edge_fill(quad, 'tr', 't')
            outer_edge_fill(quad, 'tr', 'tr')
            outer_edge_fill(quad, 'tr', 'r')
            # bl neighbors
            qbl[['t']] = qtl
            qbl[['tr']] = qtr
            qbl[['r']] = qbr
            outer_edge_fill(quad, 'bl', 'b')
            outer_edge_fill(quad, 'bl', 'bl')
            outer_edge_fill(quad, 'bl', 'l')
            # br neighbors
            qbr[['l']] = qbl
            qbr[['tl']] = qtl
            qbr[['t']] = qtr
            outer_edge_fill(quad, 'br', 'b')
            outer_edge_fill(quad, 'br', 'br')
            outer_edge_fill(quad, 'br', 'r')

            for quad_loc in [qtl, qtr, qbl, qbr]:
                dfs(quad_loc)
        
        if root is None:
            root = self.quadtree.root

        dfs(root)

    def __build_graph__(self, leaves:Optional[List[QuadNode]] = None, overwrite_directed=True):

        def recursive_search(shallow_neigh:QuadNode, directions:List[str] = ['tl']) -> List[QuadNode]:
            if shallow_neigh is None: return []
            if shallow_neigh.leaf: return [shallow_neigh]
            neigh_list = []

            for direction in directions:
                neigh_list.extend(recursive_search(shallow_neigh=shallow_neigh[direction],
                                                   directions=directions))

            return neigh_list

        if leaves is None:
            leaves = self.quadtree.leaves
        for quad in leaves:
            for neigh_str in ['tl', 't', 'tr', 'r', 'br', 'b', 'bl', 'l']:
                adjacent_neighs = recursive_search(quad[[neigh_str]][0], self.NeighOuterEdges[neigh_str])
                self.add_one_to_many(quad, adjacent_neighs, overwrite_directed=overwrite_directed)

    def build(self):
        self.__fill_shallow_neighs__()
        self.__build_graph__()

    def calculate_distance(self, node_from:QuadNode, node_to:QuadNode, scale_tranform:FieldScaleTransform=lambda x: 1.0 + x, scaled=True):
        if scaled:
            multipler = np.Inf if node_to.boundary_zone == 0 else scale_tranform(node_to.boundary_max_range)
        else:
            multipler = 1.0

        diff = node_to.center_point - node_from.center_point
        return multipler*np.linalg.norm(diff)

    def __reconstruct_path__(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]
    
    def find_path(self, start_node:QuadNode, end_node:QuadNode, scale_tranform:FieldScaleTransform=lambda x: 1.0 + x, alg:RoutingAlgorithmStr='A*', **kwargs):
        """Routing Algorithms

        Options:
        * A*
        * Dijkstra
        * [Any algorithm included by user]
        """

        return self.routing_algs[alg.lower()](start_node=start_node, end_node=end_node, scale_tranform=scale_tranform, **kwargs)
    
    def find_path_A_star(self, start_node:QuadNode, end_node:QuadNode, scale_tranform:FieldScaleTransform=lambda x: 1.0 + x) -> Optional[List[QuadNode]]:
        """ find path (A*) 
        
        Returns None if no path found
        """
        open_set = []
        heapq.heappush(open_set, (0, start_node))

        came_from = {}
        g_score = defaultdict(lambda: np.Inf)
        g_score[start_node] = 0

        f_score = defaultdict(lambda: np.Inf)
        f_score[start_node] = self.calculate_distance(start_node, end_node, scaled=False)

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end_node:
                return self.__reconstruct_path__(came_from, current)

            for neighbor in self._graph[current]:
                tentative_g_score = g_score[current] + self.calculate_distance(current, neighbor, scale_tranform=scale_tranform)

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.calculate_distance(neighbor, end_node, scaled=False)

                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None
    
    def find_path_dijkstra(self, start_node:QuadNode, end_node:QuadNode, scale_tranform:FieldScaleTransform=lambda x: 1.0 + x)-> Optional[List[QuadNode]]:
        """ find path (Dijkstra) 
        
        Returns None if no path found
        """
        open_set = []
        heapq.heappush(open_set, (0, start_node))

        came_from = {}
        dist = defaultdict(lambda: np.Inf)
        dist[start_node] = 0

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end_node:
                return self.__reconstruct_path__(came_from, current)

            for neighbor in self._graph[current]:
                tentative_dist = dist[current] + self.calculate_distance(current, neighbor, scale_tranform=scale_tranform)

                if tentative_dist < dist[neighbor]:
                    came_from[neighbor] = current
                    dist[neighbor] = tentative_dist

                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (dist[neighbor], neighbor))

        return None

    def find_route(self, pointA:Point, pointB:Point, scale_tranform:FieldScaleTransform=lambda x: 1.0 + x, alg:RoutingAlgorithmStr='A*'):
        
        quads = self.quadtree.find_quads([pointA, pointB])
        return self.find_path(quads[0], quads[1], scale_tranform=scale_tranform, alg=alg)

    def find_many_routes(self, pointsA:Point, pointsB:Point, scale_tranform:FieldScaleTransform=lambda x: 1.0 + x, alg:RoutingAlgorithmStr='A*'):
        pointsA, pointsB = np.array(pointsA), np.array(pointsB)
        n = len(pointsA)

        quads = self.quadtree.find_quads(np.concatenate([pointsA, pointsB], axis=0))

        return [self.find_path(quads[idx], quads[n+idx], scale_tranform=scale_tranform, alg=alg) for idx in range(n)]
    
    def to_routes_lines_collection(self):
        
        lines = []
        for leaf in self._graph.keys():
            for neigh in self._graph[leaf]:
                lines.extend([[leaf.center_point[0], neigh.center_point[0]], [leaf.center_point[1], neigh.center_point[1]]])

        return lines
    
    @staticmethod
    def route_to_lines_collection(pointsA:Point, pointsB:Point, route: List[QuadNode], remapped=False) -> np.ndarray:
        """
        Given a route (i.e. list of quads) and starting and ending location, it returns path of the route
        """
        pointsA = np.array(pointsA)
        pointsB = np.array(pointsB)

        route = route[1:-1] if len(route) > 1 else []

        lines = np.array([quad_stop.center_point for quad_stop in route])
        if remapped:
            if len(lines):
                lines = np.concatenate([pointsA.reshape(-1, 2), lines, pointsB.reshape(-1, 2)], axis=0)
            else:
                lines = np.concatenate([pointsA.reshape(-1, 2), pointsB.reshape(-1, 2)], axis=0)

        return lines




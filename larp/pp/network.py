from collections import defaultdict
import heapq
from typing import Callable, List, Optional, Set

import numpy as np

from larp.quad import QuadNode, QuadTree
from larp.types import Scaler, Point, RoutingAlgorithmStr

"""
Author: Josue N Rivera
"""

RoutingAlgorithm = Callable[[Point, Point, Scaler, dict], Optional[List[Point]]]

class Network(object):
    """ Network data structure, undirected by default. 
    """

    def __init__(self, connections:List[tuple] = None, directed = False):
        self._graph = defaultdict(set)
        self._directed = directed
        if connections is not None:
            self.add_connections(connections)

    def __getitem__(self, node):
        """
        Returns neighbors of a node
        """
        return self._graph[node]

    def add_connections(self, connections:List[tuple]):
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

        for _, cxns in self._graph.items():  # python3: items()
            try:
                cxns.remove(node)
            except KeyError:
                pass
        
        self._graph.pop(node)

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """
        return node1 in self._graph and node2 in self._graph[node1]

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
    
class QuadNetwork(Network):

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

    def __init__(self, quadtree:QuadTree, directed:bool=False, build_network:bool=True):
        self.quadtree = quadtree
        super().__init__(directed=directed)

        if build_network:
            self.build()

    def __fill_shallow_neighs__(self, root:Optional[QuadNode] = None):
        """
        Populates shallow (immediate) neighbor relationships between quads in the quadtree.

        For each non-leaf node, this function connects each of its four children ('tl', 'tr', 'bl', 'br')
        to their internal siblings (in the same parent) and to appropriate neighboring child nodes in 
        adjacent quads.

        Args:
            root (Optional[QuadNode]): The root node to start from. Defaults to the quadtree root.
        """

        if root is None:
            root = self.quadtree.root

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

        dfs(root)

    def __build_graph__(self, leaves:Optional[List[QuadNode]] = None, overwrite_directed=True):
        """
        Constructs the connectivity graph of the leaf quads in the quadtree.

        Each leaf quad is connected to all of its reachable neighbors across outer quad boundaries.
        The traversal for neighbors uses predefined directional mappings to ensure complete adjacency.

        Args:
            leaves (Optional[List[QuadNode]]): List of leaves to include in the graph.
                                            If None, all leaves from the quadtree are used.
            overwrite_directed (bool): Whether to overwrite existing directed edges in the network.
                                    This affects how bidirectional graphs behave.
        """

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

    def build(self, root:Optional[QuadNode] = None):
        """
        Builds the quad network by establishing neighbor links and generating the connectivity graph.

        If no root is provided, the entire quadtree is processed. If a root node is specified,
        only the subnetwork rooted at that node is built.

        Args:
            root (Optional[QuadNode]): The root node to build from. If None, builds the entire network.
        """
        
        if root is None:
            self.__fill_shallow_neighs__()
            self.__build_graph__()
        else:
            self.__fill_shallow_neighs__(root)
            self.__build_graph__(leaves=self.quadtree.search_leaves(root), overwrite_directed=False)

    def refresh(self, full_rebuild=False):
        """
        Refreshes the quad network after the quadtree has changed.

        - Removes references to obsolete nodes.
        - Adds new nodes.
        - Rebuilds shallow neighbors and connections for the new nodes.
        """
        if full_rebuild:
            self._graph.clear()
            self.build()
        else:
            current_nodes = set(self._graph.keys())
            new_leaves = set(self.quadtree.leaves)

            # Nodes to remove (no longer leaves)
            removed_nodes = current_nodes - new_leaves
            for node in removed_nodes:
                self.remove(node)  # Also removes node from neighbors

            # Nodes to add (new leaves not already in graph)
            added_nodes = new_leaves - current_nodes

            if added_nodes:
                # Fill shallow neighbors before building connections
                self.__fill_shallow_neighs__()
                self.__build_graph__(leaves=list(added_nodes), overwrite_directed=False)

    def find_quad(self, points:List[Point]):
        return self.quadtree.find_quad(points)
    
    def get_quad_center(self, quads:List[QuadNode]):
        return [quad.center_point for quad in quads]
        
    def get_quad_size(self, quads:List[QuadNode]):
        return [quad.size for quad in quads]
    
    def get_shared_entry_point(self, node_from: QuadNode, node_to: QuadNode, entry: Point) -> np.ndarray:
        """
        Computes the entry point on the shared edge or corner with a neighbor quad.
        If diagonally adjacent, returns midpoint of shared corner.
        """

        shared_edge = node_from.get_shared_edge(node_to)

        if shared_edge is None:
            return None

        p0, p1 = shared_edge

        if np.allclose(p0, p1):
            # Shared corner: return the corner point
            return p0
        else:
            # Shared edge: project entry point onto the edge segment
            edge_vector = p1 - p0
            edge_length_squared = np.dot(edge_vector, edge_vector)
            if edge_length_squared == 0:
                # Avoid division by zero if edge_vector is zero
                return p0
            t = np.dot(entry - p0, edge_vector) / edge_length_squared
            t = np.clip(t, 0.0, 1.0)
            projection = p0 + t * edge_vector
            return projection
    
    def get_corner_points(self, quad_path: List[QuadNode]):

        i = 0
        points = []
        for i in range(len(quad_path)-1):
            shared_edge = quad_path[i].get_shared_edge(quad_path[i+1])

            if shared_edge is None:
                return None
            
            if np.allclose(shared_edge[0], shared_edge[1]):
                points.append(shared_edge[0])

        return np.array(points)

    def get_center_distance(self, node_from:QuadNode, node_to:QuadNode, scaler:Optional[Scaler]=None, max_scale: float = np.inf):

        if scaler is None:
            multipler = 1.0
        else:
            multipler = np.inf if node_to.boundary_zone == 0 else scaler(node_to.boundary_max_range)
        
        multipler = min(multipler, max_scale)

        diff = node_to.center_point - node_from.center_point
        return multipler*np.linalg.norm(diff)
    
    def to_line_collection(self):
        """
        Generates a list of 2D line segments for use with matplotlib's LineCollection.

        Returns:
            List of 2-point arrays: [[(x1, y1), (x2, y2)], ...]
        """
        
        lines = []
        for leaf in self._graph.keys():
            for neigh in self[leaf]:
                lines.append([leaf.center_point, neigh.center_point])

        return np.array(lines)
    
from collections import defaultdict
from typing import List

import numpy as np

from larp.quad import QuadNode, QuadTree
from larp.types import Point

"""
Author: Josue N Rivera
"""

class Graph(object):
    """ Graph data structure, undirected by default. 
    
    - Adapted from: https://stackoverflow.com/questions/19472530/representing-graphs-data-structure-in-python
    """

    def __init__(self, connections = None, directed = False):
        self._graph = defaultdict(set)
        self._directed = directed
        if connections is not None:
            self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """

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
                self.add(node1, node2)

    def remove(self, node):
        """ Remove all references to node """

        for n, cxns in self._graph.items():  # python3: items(); python2: iteritems()
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

    def find_path(self, node1, node2, path=[]):
        """ A* find shortest path"""

        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
    
class RouteGraph(Graph):

    NeighOuterEdges = { # Maps child quad' outer edge to neigbors children
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

    def __init__(self, quad_tree:QuadTree, directed:bool=False, build_graph:bool=True):
        self.quad_tree = quad_tree
        super.__init__(directed=directed)

        if build_graph:
            self.build()

    def __fill_shallow_neighs__(self):

        def outer_edge_fill(quad:QuadNode, child = 'tl', side = 't'):
            child_quad = quad[child]
            parent_neigh:QuadNode = quad[[side]][0]

            subs = self.NeighOuterEdges[child][side]
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
            outer_edge_fill(quad, 'bl', 'r')
            # br neighbors
            qbr[['l']] = qbl
            qbr[['tl']] = qtl
            qbr[['t']] = qtr
            outer_edge_fill(quad, 'br', 'b')
            outer_edge_fill(quad, 'br', 'br')
            outer_edge_fill(quad, 'br', 'r')

            for quad_loc in [qtl, qtr, qbl, qbr]:
                dfs(quad_loc)
        
        dfs(self.quad_tree.root)

    def __build_graph__(self):

        def recursive_search(shallow_neigh:QuadNode, neigh_list:List[QuadNode], direction:str = 't'):
            if shallow_neigh.leaf: return shallow_neigh

            neigh_list.extend(recursive_search(shallow_neigh=shallow_neigh[direction],
                                               neigh_list=neigh_list,
                                               direction=direction))
            if len(direction) < 2:
                neigh_list.extend(recursive_search(shallow_neigh=shallow_neigh[direction],
                                                neigh_list=neigh_list,
                                                direction=direction))

            return neigh_list

        def adjacent_neighs_search(quad:QuadNode) -> List[QuadNode]:
            out = []

            for neigh_str in ['tl', 't', 'tr', 'r', 'br', 'b', 'bl', 'l']:
                out.extend(recursive_search(quad[[neigh_str]], [], neigh_str))

            return out


        for quad in self.quad_tree.leaves:
            adjacent_neighs = adjacent_neighs_search(quad=quad)
            self.add_one_to_many(quad, adjacent_neighs, overwrite_directed=True)

    def build(self):
        self.__fill_shallow_neighs__()
        self.__build_graph__()

    def find_route(self, pointA:Point, pointB:Point):
        
        quads = self.quad_tree.find_quads([pointA, pointB])
        return self.find_path(quads[0], quads[1])

    def find_many_routes(self, pointsA:np.ndarray, pointsB:np.ndarray):
        pointsA, pointsB = np.array(pointsA), np.ndarray(pointsB)
        n = len(pointsA)

        quads = self.quad_tree.find_quads(np.concatenate([pointsA, pointsB], axis=0))

        return [self.find_path(quads[idx], quads[n+idx]) for idx in range(n)]



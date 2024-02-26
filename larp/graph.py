from collections import defaultdict

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

    def __init__(self, quad_tree:QuadTree, directed=False):
        self.quad_tree = quad_tree
        super.__init__(directed=directed)

    def __fill_shallow_neighs__(self):

        def side_fill(quad, corner = 'tl', idx = ['t', 'tr', 'bl', 'br']):

            qc = quad['tl']
            nt = quad[['t']][0]
            if nt is not None or nt.leaf:
                qc[['t', 'tr']] = nt
            else:
                qc[['t']] = nt['bl']
                qc[['tr']] = nt['br']

        def dfs(quad:QuadNode):
            if quad.leaf: return

            qtl, qtr, qbl, qbr = quad['tl'], quad['tr'], quad['bl'], quad['br']

            # fill tl neighbors
            qtl[['r']] = qtr
            qtl[['br']] = qbr
            qtl[['b']] = qbl

            nt = quad[['t']][0]
            if nt is not None or nt.leaf:
                qtl[['t', 'tr']] = nt
            else:
                qtl[['t']] = nt['bl']
                qtl[['tr']] = nt['br']

            for quad_loc in [qtl, qtr, qbl, qbr]:
                dfs(quad_loc)
        
        dfs(self.quad_tree.root)

    def __build_graph__(self):
        pass

    def build(self):
        self.__fill_shallow_neighs__()
        self.__build_graph__()

    def __fill_shallow_neighs__(self):

        def side_fill(quad, corner = 'tl', idx = ['t', 'tr', 'bl', 'br']):

            qc = quad[corner]
            nt = quad[['t']][0]
            if nt is not None or nt.leaf:
                qc[['t', 'tr']] = nt
            else:
                qc[['t']] = nt['bl']
                qc[['tr']] = nt['br']

        def dfs(quad:QuadNode):
            if quad.leaf: return

            qtl, qtr, qbl, qbr = quad['tl'], quad['tr'], quad['bl'], quad['br']

            # fill tl neighbors
            qtl[['r']] = qtr
            qtl[['br']] = qbr
            qtl[['b']] = qbl

            nt = quad[['t']][0]
            if nt is not None or nt.leaf:
                qtl[['t', 'tr']] = nt
            else:
                qtl[['t']] = nt['bl']
                qtl[['tr']] = nt['br']

            for quad_loc in [qtl, qtr, qbl, qbr]:
                dfs(quad_loc)
        
        dfs(self.quad_tree.root)

    def __build_graph__(self):
        pass

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



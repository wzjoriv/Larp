
from larp.field import PotentialField
from larp.quad import QuadTree
from larp.graph import RouteGraph

"""
Author: Josue N Rivera
"""

class HotReloader(object):

    def __init__(self, field:PotentialField, quadtree:QuadTree, graph:RouteGraph):
        self.field = field
        self.quadtree = quadtree
        self.graph = graph

    def addField(self, new_field:PotentialField):
        pass

    def removeRGJ(self, rgj):
        pass


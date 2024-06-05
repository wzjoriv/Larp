
from larp.field import PotentialField
from larp.quad import QuadTree
from larp.graph import RouteGraph

"""
Author: Josue N Rivera
"""

class Reloader(object):

    def __init__(self, field:PotentialField, tree:QuadTree, graph:RouteGraph):
        self.field = field
        self.tree = tree
        self.graph = graph

    def addField(self, new_field):
        pass


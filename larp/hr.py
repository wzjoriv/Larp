
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
        
        new_field.reload_center_point(False)
        new_field.center_point = self.field.center_point
        new_field.size = self.field.size

        new_quadtree = QuadTree(new_field,
                                minimum_sector_length=self.quadtree.min_sector_size,
                                maximum_sector_length=self.quadtree.max_sector_size,
                                boundaries=self.quadtree.boundaries,
                                size=self.quadtree.size,
                                build_tree=True)

    def removeRGJ(self, idx):
        rgj = self.field[idx]

        search_field = PotentialField(rgj)
        search_field.reload_center_point(False)
        search_field.center_point = self.field.center_point
        search_field.size = self.field.size
        search_quadtree = QuadTree(search_field,
                                minimum_sector_length=self.quadtree.min_sector_size,
                                maximum_sector_length=self.quadtree.max_sector_size,
                                boundaries=self.quadtree.boundaries,
                                size=self.quadtree.size,
                                build_tree=True)
        
        self.field.delRGJ(idx=idx)


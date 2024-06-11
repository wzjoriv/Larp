
import numpy as np
from larp.field import PotentialField, RGJGeometry
from larp.quad import QuadTree, QuadNode
from larp.graph import RouteGraph

"""
Author: Josue N Rivera
"""

class HotLoader(object):

    def __init__(self, field:PotentialField, quadtree:QuadTree, graph:RouteGraph):
        self.field = field
        self.quadtree = quadtree
        self.graph = graph

    def addField(self, new_field:PotentialField) -> None:
        
        new_field.reload_center_point(False)
        new_field.center_point = self.field.center_point
        new_field.size = self.field.size
        new_qtree = QuadTree(new_field,
                             minimum_sector_length=self.quadtree.min_sector_size,
                             maximum_sector_length=self.quadtree.max_sector_size,
                             boundaries=self.quadtree.boundaries,
                             size=self.quadtree.size,
                             build_tree=True)
        
        # Add rgj to field
        for rgj in new_field:
            self.field.addRGJ(rgj=rgj)
        
        # Update field idx in new quadtree
        def update_idx(quad:QuadNode):
            if quad is None:
                return
            quad.rgj_idx = quad.rgj_idx+self.quadtree.n_zones
            for child in quad.children:
                update_idx(child)
        
        update_idx(new_qtree.root)

        # Update quadtree
        graph_active_quad = set()
        def update_quad(rootquad:QuadNode, newquad:QuadNode) -> bool:
            """
            Returns whether to update tree or not
            """
            
            # Don't bother updating quad far from restrictions (Note: newquad will be leaf if true)
            if newquad is None or newquad.boundary_zone == self.quadtree.n_zones: 
                return False
            
            if rootquad is None:
                return True
            
            # update info
            if newquad.boundary_zone < rootquad.boundary_zone:
                rootquad.boundary_zone = newquad.boundary_zone
            rootquad.rgj_idx = np.append(rootquad.rgj_idx, newquad.rgj_idx)
            rootquad.rgj_zones = np.append(rootquad.rgj_zones, newquad.rgj_zones)

            if rootquad.leaf and not newquad.leaf:
                return True

            for child in ['tl', 'tr', 'bl', 'br']:
                if update_quad(rootquad[child], newquad[child]):
                    # mark quad to update in graph
                    graph_active_quad.add(rootquad)

                    # TODO: if original root child quad has rgjs (idx < n_zones), build quad with own rgjs. Else do below
                    # replace rest of old branch with new branch
                    if rootquad[child] is not None and rootquad[child].leaf:
                        self.quadtree.leaves.remove(rootquad[child])
                    
                    rootquad[child] = newquad[child]
                    self.quadtree.leaves.update(self.quadtree.search_leaves(rootquad[child]))

            return False

        update_quad(self.quadtree.root, new_qtree.root)


        # TODO: update graph (recalculate neighbors for active quad)
        # Delete node in active quad

    def addRGJ(self, rgj:RGJGeometry) -> int:
        """
        Returns index of added rgj
        """

        self.addField(PotentialField(rgj))
        return len(self.field) - 1

    def removeRGJ(self, idx:int):
        rgj = self.field[idx]

        search_field = PotentialField(rgj)
        search_field.reload_center_point(False)
        search_field.center_point = self.field.center_point
        search_field.size = self.field.size
        search_qtree = QuadTree(search_field,
                                minimum_sector_length=self.quadtree.min_sector_size,
                                maximum_sector_length=self.quadtree.max_sector_size,
                                boundaries=self.quadtree.boundaries,
                                size=self.quadtree.size,
                                build_tree=True)
        
        # update quadtree
        
        self.field.delRGJ(idx=idx)


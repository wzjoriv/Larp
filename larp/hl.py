
from typing import List, Union
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

    def addField(self, new_field:PotentialField) -> List[int]:
        
        new_field.reload_center_point(False)
        new_field.center_point = self.field.center_point
        new_field.size = self.field.size
        new_qtree = QuadTree(new_field,
                             minimum_sector_length=self.quadtree.min_sector_size,
                             boundaries=self.quadtree.boundaries,
                             size=self.quadtree.size,
                             build_tree=True)
        
        n_original = len(self.field)
        
        # Add rgj to field
        for rgj in new_field:
            self.field.addRGJ(rgj=rgj)
        
        # Update field idx in new quadtree
        def update_idx(quad:QuadNode):
            if quad is None:
                return
            quad.rgj_idx = quad.rgj_idx+n_original
            for child in quad.children:
                update_idx(child)
        
        # At the beginning because it is used by all. Needed by quad build and rgj_idx update
        update_idx(new_qtree.root)

        # Update quadtree
        graph_active_quad_new = set()
        graph_active_quad_old = set()

        def replace_branch(rootquad, newquad, child):
            rootquad[child] = newquad[child]
            new_leaves = set(self.quadtree.search_leaves(rootquad[child]))
            self.quadtree.leaves.update(new_leaves)
            graph_active_quad_new.update(new_leaves) # mark quad to update in graph

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

                    if rootquad[child] is not None:
                        old_leaves = set(self.quadtree.search_leaves(rootquad[child]))
                        self.quadtree.leaves = self.quadtree.leaves - old_leaves
                        graph_active_quad_old.update(old_leaves) # mark quad to update in graph

                        if (rootquad[child].rgj_idx < n_original).any(): # if original branch has rgjs
                            rootquad[child] = self.quadtree.__build__(rootquad[child].center_point,
                                                                    rootquad[child].size,
                                                                    filter_idx=rootquad[child].rgj_idx,
                                                                    aggressive=True)
                            new_leaves = set(self.quadtree.search_leaves(rootquad[child]))
                            graph_active_quad_new.update(new_leaves)
                        else:
                            replace_branch(rootquad, newquad, child)
                    else:
                        replace_branch(rootquad, newquad, child)

            return False

        update_quad(self.quadtree.root, new_qtree.root)

        # delete old reference
        for node in graph_active_quad_old:
            self.graph.remove(node)
            del node

        # add new references
        self.graph.__fill_shallow_neighs__()
        self.graph.__build_graph__(graph_active_quad_new, overwrite_directed=False)

        return np.arange(n_original, len(self.field))

    def addRGJ(self, rgj:RGJGeometry) -> int:
        """
        Returns index of added rgj
        """

        return self.addField(PotentialField([rgj]))[0]

    def removeRGJ(self, idxs:Union[int, List[int]]) -> None:

        idxs = np.unique([idxs] if idxs is int else idxs)
        rgjs = [self.field.rgjs[idx] for idx in idxs]

        search_field = PotentialField(rgjs)
        search_field.reload_center_point(False)
        search_field.center_point = self.field.center_point
        search_field.size = self.field.size
        search_qtree = QuadTree(search_field,
                                minimum_sector_length=self.quadtree.min_sector_size,
                                boundaries=self.quadtree.boundaries,
                                size=self.quadtree.size,
                                build_tree=True)
        
        # Remove rgj from field
        self.field.delRGJ(idxs)
        
        # update quadtree
        graph_active_quad_new = set()
        graph_active_quad_old = set()

        def update_quad(rootquad:QuadNode, delquad:QuadNode):
            """
            Returns whether to consider merge quads or not 
            """

            if delquad is None:
                return False
            
            # update info (remove idxs)
            mask = ~np.in1d(rootquad.rgj_idx, idxs[delquad.rgj_idx])
            rootquad.rgj_idx = rootquad.rgj_idx[mask]
            rootquad.rgj_zones = rootquad.rgj_zones[mask]
            if delquad.boundary_zone == rootquad.boundary_zone:
                rootquad.boundary_zone = min(rootquad.rgj_zones) if len(rootquad.rgj_idx) > 0 else self.quadtree.n_zones

            # Update indexes in quad
            original_idx = rootquad.rgj_idx.copy()
            for idx in idxs:
                mask = original_idx >= idx
                rootquad.rgj_idx[mask] = rootquad.rgj_idx[mask] - 1
            
            #If all true, consider merging smaller quad
            if all([update_quad(rootquad[child], delquad[child]) for child in ['tl', 'tr', 'bl', 'br']]):

                #If maximum size not violated, merge
                if rootquad.size <= self.quadtree.max_sector_size and \
                (rootquad['tl'].boundary_zone == rootquad['tr'].boundary_zone == rootquad['bl'].boundary_zone == rootquad['br'].boundary_zone):
                    old_leaves = set(self.quadtree.search_leaves(rootquad))
                    self.quadtree.leaves = self.quadtree.leaves - old_leaves
                    graph_active_quad_old.update(old_leaves) # mark quad to update in graph

                    del rootquad.children
                    rootquad.children = [None]*len(rootquad.chdToIdx)
                    self.quadtree.mark_leaf(rootquad)

                    graph_active_quad_new.add(rootquad) # mark quad to update in graph
                    return True
                
            if not len(rootquad.rgj_idx): 
                return True

            return False

        update_quad(self.quadtree.root, search_qtree.root)

        # delete old reference
        for node in graph_active_quad_old:
           self.graph.remove(node)
           del node

        # add new references
        self.graph.__fill_shallow_neighs__()
        self.graph.__build_graph__(graph_active_quad_new, overwrite_directed=False)


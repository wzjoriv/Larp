import numpy as np
import sys
sys.path.append("../larp")
import larp

"""
Author: Josue N Rivera

"""

def test_add_remove_field():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 60], 
        'repulsion': [[5, 0], [0, 5]]
    }]

    field = larp.PotentialField(size=50, center_point=[55, 55], rgjs=point_rgjs)
    quadtree = larp.quad.QuadTree(field=field,
                                  minimum_length_limit=5,
                                  edge_bounds=np.arange(0.2, 0.8, 0.2),
                                  build_tree=True)
    network = larp.network.RoutingNetwork(quadtree=quadtree)
    
    loader = larp.hl.HotLoader(field=field, quadtree=quadtree, network=network)
    added_idx = loader.addRGJ(larp.PointRGJ((55, 55), repulsion=[[10, 0], [0, 10]]))

    loader.removeRGJ(added_idx)

test_add_remove_field()

def test_add_rgj_idx_passed():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 60], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 50], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [50, 60], 
        'repulsion': [[5, 0], [0, 5]]
    }]

    field = larp.PotentialField(size=40, center_point=[55, 55], rgjs=point_rgjs)
    quadtree = larp.quad.QuadTree(field=field,
                                  minimum_length_limit=5,
                                  edge_bounds=np.arange(0.2, 0.8, 0.2),
                                  build_tree=True)
    network = larp.network.RoutingNetwork(quadtree=quadtree)
    
    loader = larp.hl.HotLoader(field=field, quadtree=quadtree, network=network)
    loader.addRGJ(larp.PointRGJ((55, 55), repulsion=[[25, 0], [0, 25]]))

    def get_rgj_idx(quad:larp.quad.QuadNode):
        if quad.leaf:
            return
        
        for child in quad.children:
            assert set(child.rgj_idx) <= set(quad.rgj_idx), f"Child [{str(child)}] rgj idxs are not a subset of {str(quad)}'s"
            assert child.boundary_zone >= quad.boundary_zone, f"Child {str(child)} boundary zone is higher than {str(quad)}'s"

        for child in quad.children:
            get_rgj_idx(child)

    get_rgj_idx(quadtree.root)

test_add_rgj_idx_passed()

def test_remove_rgj_idx_passed():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 60], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 50], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [50, 60], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [55, 55], 
        'repulsion': [[25, 0], [0, 25]]
    }]

    field = larp.PotentialField(size=40, center_point=[55, 55], rgjs=point_rgjs)
    quadtree = larp.quad.QuadTree(field=field,
                                  minimum_length_limit=0.5,
                                  edge_bounds=np.arange(0.2, 0.8, 0.2),
                                  build_tree=True)
    network = larp.network.RoutingNetwork(quadtree=quadtree)
    
    loader = larp.hl.HotLoader(field=field, quadtree=quadtree, network=network)
    loader.removeRGJ([2, 3])

    def get_rgj_idx(quad:larp.quad.QuadNode):
        if quad.leaf:
            return
        
        for child in quad.children:
            assert child.boundary_zone >= quad.boundary_zone, f"Child {str(child)} boundary zone is higher than {str(quad)}'s"
            assert set(child.rgj_idx) <= set(quad.rgj_idx), f"Child {str(child)} rgj idxs are not a subset of {str(quad)}'s"

        for child in quad.children:
            get_rgj_idx(child)
        
    get_rgj_idx(quadtree.root)

test_remove_rgj_idx_passed()

def test_add_leaf_none_children():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 60], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 50], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [50, 60], 
        'repulsion': [[5, 0], [0, 5]]
    }]

    field = larp.PotentialField(size=40, center_point=[55, 55], rgjs=point_rgjs)
    quadtree = larp.quad.QuadTree(field=field,
                                  minimum_length_limit=5,
                                  edge_bounds=np.arange(0.2, 0.8, 0.2),
                                  build_tree=True)
    network = larp.network.RoutingNetwork(quadtree=quadtree)
    
    loader = larp.hl.HotLoader(field=field, quadtree=quadtree, network=network)
    loader.addRGJ(larp.PointRGJ((55, 55), repulsion=[[25, 0], [0, 25]]))

    for quad in quadtree.leaves:
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by list) with non-none children"

    for quad in quadtree.search_leaves():
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by search) with non-none children"

    assert quadtree.search_leaves() == quadtree.leaves, "Leaves in list are different than those found by search"

test_add_leaf_none_children()

def test_remove_leaf_none_children():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 60], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [60, 50], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [50, 60], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [55, 55], 
        'repulsion': [[25, 0], [0, 25]]
    }]

    field = larp.PotentialField(size=40, center_point=[55, 55], rgjs=point_rgjs)
    quadtree = larp.quad.QuadTree(field=field,
                                  minimum_length_limit=1,
                                  edge_bounds=np.arange(0.2, 0.8, 0.2),
                                  build_tree=True)
    network = larp.network.RoutingNetwork(quadtree=quadtree)
    
    loader = larp.hl.HotLoader(field=field, quadtree=quadtree, network=network)
    loader.removeRGJ([2, 3])

    for quad in quadtree.leaves:
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by list) with non-none children"

    for quad in quadtree.search_leaves():
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by search) with non-none children"

    assert quadtree.search_leaves() == quadtree.leaves, "Leaves in list are different than those found by search"

test_remove_leaf_none_children()
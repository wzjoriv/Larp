import numpy as np
import sys
sys.path.append("../larp")
import larp

"""
Author: Josue N Rivera

"""

def test_quad_on_simple_pf():
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
                                  build_tree=True,
                                  minimum_sector_length=2,
                                  maximum_sector_length=5,
                                  boundaries=np.arange(0.2, 0.8, 0.2))
    
    quadtree.build()

    aleaf = next(iter(quadtree.leaves))
    assert aleaf == quadtree.find_quads([aleaf.center_point])[0], "Leaf found does not match expected leaf"

    manyleaves = list(quadtree.leaves)[:30]
    quads = quadtree.find_quads([quad.center_point for quad in manyleaves])
    assert np.array([manyleaves[idx] == quads[idx] for idx in range(len(quads))]).all(), "Many quads search failed"

    assert quadtree.leaves == set(quadtree.search_leaves(quadtree.root)), "Leaves stored and leaves searched don't match"

test_quad_on_simple_pf()

def test_rgj_idx_passed():
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
                                  minimum_sector_length=5,
                                  boundaries=np.arange(0.2, 0.8, 0.2),
                                  build_tree=True)

    def get_rgj_idx(quad:larp.quad.QuadNode):

        if quad.leaf:
            return
        
        for child in quad.children:
            assert set(child.rgj_idx) <= set(quad.rgj_idx), f"Child {str(child)} rgj idxs are not a subset of {str(quad)}'s"
            assert child.boundary_zone >= quad.boundary_zone, f"Child {str(child)} boundary zone is higher than {str(quad)}'s"

        for child in quad.children:
            get_rgj_idx(child)

    get_rgj_idx(quadtree.root)

test_rgj_idx_passed()

def test_leaf_none_children():
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
                                  minimum_sector_length=5,
                                  boundaries=np.arange(0.2, 0.8, 0.2),
                                  build_tree=True)

    for quad in quadtree.leaves:
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (in list) with non-none children"

    for quad in quadtree.search_leaves():
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by search) with non-none children"

    assert quadtree.search_leaves() == quadtree.leaves, "Leaves in list are different than those found by search"

test_leaf_none_children()

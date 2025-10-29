import numpy as np
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
                                  minimum_length_limit=2,
                                  maximum_length_limit=5,
                                  edge_bounds=np.arange(0.2, 0.8, 0.2))
    
    quadtree.build()

    aleaf = next(iter(quadtree.leaves))
    assert aleaf == quadtree.find_quad([aleaf.center_point])[0], "Leaf found does not match expected leaf"

    manyleaves = list(quadtree.leaves)[:30]
    quads = quadtree.find_quad([quad.center_point for quad in manyleaves])
    assert np.array([manyleaves[idx] == quads[idx] for idx in range(len(quads))]).all(), "Many quads search failed"

    assert quadtree.leaves == set(quadtree.search_leaves(quadtree.root)), "Leaves stored and leaves searched don't match"

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
                                  minimum_length_limit=5,
                                  edge_bounds=np.arange(0.2, 0.8, 0.2),
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
                                  minimum_length_limit=5,
                                  edge_bounds=np.arange(0.2, 0.8, 0.2),
                                  build_tree=True)

    for quad in quadtree.leaves:
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (in list) with non-none children"

    for quad in quadtree.search_leaves():
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by search) with non-none children"

    assert quadtree.search_leaves() == quadtree.leaves, "Leaves in list are different than those found by search"

def test_iter_quadtree():

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
    
    for quad in quadtree:
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (in list) with non-none children"

    for quad in quadtree.search_leaves():
        assert all([child is None for child in quad.children]) == True, f"{str(quad)} is leaf (by search) with non-none children"

    assert quadtree.search_leaves() == quadtree.leaves, "Leaves in list are different than those found by search"

def test_quad_link():

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
    
    quad_chain = quadtree.find_quads_chain([[50.1, 50.1]])[0]

    assert len(quad_chain) == 4, "The expected number of quads in the chain is different than the number found"

    assert all([quad.boundary_zone == 0 for quad in quad_chain]), "The expected number of quads in the chain is different than the number found"

def test_boundary():

    quad = larp.quad.QuadNode((2, 3), 2)
    assert np.allclose(quad.get_boundaries(), np.array([1, 2, 3, 4])) , "Expected edge of quads not returned"

def test_quad_shared_edge():

    # Scenerio 1: Sharing partial edge
    quad1 = larp.quad.QuadNode((2, 3), 2)
    quad2 = larp.quad.QuadNode((4, 2), 2)
    shared_edge1 = quad1.get_shared_edge(quad2)
    shared_edge2 = quad2.get_shared_edge(quad1)

    assert np.allclose(shared_edge1, np.array([[3, 2], [3, 3]])) , "Expected edge of quads not returned"
    assert np.allclose(shared_edge1, shared_edge2) , "Expected edge of quads not returned"

    # Scenerio 2: Big box to the side of small box
    quad1 = larp.quad.QuadNode((2, 4), 2)
    quad2 = larp.quad.QuadNode((5, 4), 4)
    shared_edge1 = quad1.get_shared_edge(quad2)
    shared_edge2 = quad2.get_shared_edge(quad1)

    assert np.allclose(shared_edge1, np.array([[3, 3], [3, 5]])) , "Expected edge of quads not returned"
    assert np.allclose(shared_edge1, shared_edge2) , "Expected edge of quads not returned"

    # Scenerio 3: Big box on top of small box
    quad1 = larp.quad.QuadNode((3.5, 6.5), 3)
    quad2 = larp.quad.QuadNode((4, 4), 2)
    shared_edge1 = quad1.get_shared_edge(quad2)
    shared_edge2 = quad2.get_shared_edge(quad1)

    assert np.allclose(shared_edge1, np.array([[3, 5], [5, 5]])) , "Expected edge of quads not returned"
    assert np.allclose(shared_edge1, shared_edge2) , "Expected edge of quads not returned"

    # Scenerio 4: Not sharing an edge
    quad1 = larp.quad.QuadNode((3, 14), 2)
    quad2 = larp.quad.QuadNode((6, 11), 2)
    shared_edge1 = quad1.get_shared_edge(quad2)
    shared_edge2 = quad2.get_shared_edge(quad1)

    assert shared_edge1 is None , "Edge returned when none expected"
    assert shared_edge1 is shared_edge2, "Expected edge of quads not returned"

    # Scenerio 5: Touching at a corner
    quad1 = larp.quad.QuadNode((2, 1), 2)
    quad2 = larp.quad.QuadNode((4, 3), 2)
    shared_edge1 = quad1.get_shared_edge(quad2)
    shared_edge2 = quad2.get_shared_edge(quad1)

    assert np.allclose(shared_edge1, np.array([[3, 2], [3, 2]])) , "Expected edge of quads not returned"
    assert np.allclose(shared_edge1, shared_edge2) , "Expected edge of quads not returned"

    quad1 = larp.quad.QuadNode((4, 1), 2)
    quad2 = larp.quad.QuadNode((2, 3), 2)
    shared_edge1 = quad1.get_shared_edge(quad2)
    shared_edge2 = quad2.get_shared_edge(quad1)

    assert np.allclose(shared_edge1, np.array([[3, 2], [3, 2]])) , "Expected edge of quads not returned"
    assert np.allclose(shared_edge1, shared_edge2) , "Expected edge of quads not returned"

    # Scenerio 6: Overlap
    quad1 = larp.quad.QuadNode((102, -6), 2)
    quad2 = larp.quad.QuadNode((103, -7), 2)
    shared_edge1 = quad1.get_shared_edge(quad2)
    shared_edge2 = quad2.get_shared_edge(quad1)

    assert shared_edge1 is None , "Edge returned when none expected"
    assert shared_edge1 is shared_edge2, "Expected edge of quads not returned"

def test_quad_bbox():

    # Scenerio 1: Sharing partial edge
    quad1 = larp.quad.QuadNode((2, 3), 2)
    quad2 = larp.quad.QuadNode((4, 2), 2)

    in_bbox1 = quad1.in_bbox([[1, 2], [2, 3], [-1, -1], [4, 2], [3, 3], [3.1, 2], [2.9, 2]])
    in_bbox2 = quad2.in_bbox([[1, 2], [2, 3], [-1, -1], [4, 2], [3, 3], [3.1, 2], [2.9, 2]])

    assert np.allclose(in_bbox1, np.array([True, True, False, False, True, False, True])) , "Point misclassified as being or not inside the bounding box"
    assert np.allclose(in_bbox2, np.array([False, False, False, True, True, True, False])) , "Point misclassified as being or not inside the bounding box"
    
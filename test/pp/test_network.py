import numpy as np
import sys
sys.path.append("../larp")
import larp

"""
Author: Josue N Rivera

"""

def test_network_on_simple_pf():
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
                                  minimum_length_limit=5,
                                  edge_bounds=np.arange(0.2, 0.8, 0.2))
    
    network = larp.pp.QuadNetwork(quadtree=quadtree, build_network=True)

test_network_on_simple_pf()

def test_refresh_of_network():

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
                                  minimum_length_limit=5,
                                  edge_bounds=np.arange(0.2, 0.8, 0.2))
    
    network = larp.pp.QuadNetwork(quadtree=quadtree, build_network=True)

    qfield = larp.QPotentailField(quadtree)
    qfield.delRGJ([1])
    
    network.refresh()

test_refresh_of_network()
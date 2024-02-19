import numpy as np
import sys
sys.path.append("E:\dev\Projects\LARP")

import larp

"""
Author: Josue N Rivera

TODO: Not implemneted yet
"""

def test_quad_on_simple_pf():
    point_rgjs = [{
        'type': "Point",
        'coordinates': [50, 50], 
        'repulsion': [[5, 0], [0, 5]]
    },{
        'type': "Point",
        'coordinates': [50, 50], 
        'repulsion': [[5, 0], [0, 5]]
    }]

    field = larp.PotentialField(rgjs=point_rgjs)
    quadtree = larp.quad.QuadTree(field=field)
    

test_quad_on_simple_pf()
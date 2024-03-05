import numpy as np
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
        'coordinates': [60, 60], 
        'repulsion': [[5, 0], [0, 5]]
    },
        {
            "type": "LineString",
            "coordinates": [[10, 10], [80, 10], [80, 80], [10, 80], [10, 30]], 
            "repulsion": [[9, 0], [0, 9]]
        }]

    field = larp.PotentialField(size=50, center_point=[55, 55], rgjs=point_rgjs)
    quadtree = larp.quad.QuadTree(field=field,
                                  build_tree=True,
                                  minimum_sector_length=2,
                                  maximum_sector_length=5,
                                  boundaries=np.arange(0.2, 0.8, 0.2))
    
    quadtree.build()
    

test_quad_on_simple_pf()
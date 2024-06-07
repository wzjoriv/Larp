import numpy as np
import sys
sys.path.append("../larp")
import larp

"""
Author: Josue N Rivera

"""

def test_add_field():
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
                                  minimum_sector_length=5,
                                  boundaries=np.arange(0.2, 0.8, 0.2))
    quadtree.build()
    graph = larp.graph.RouteGraph(quadtree=quadtree)
    
    hloader = larp.hr.HotReloader(field=field, quadtree=quadtree, graph=graph)
    
    
if __name__ == "__main__":
    test_add_field()
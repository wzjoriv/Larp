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
                                  minimum_sector_length=5,
                                  boundaries=np.arange(0.2, 0.8, 0.2))
    
    quadtree.build()

    aleaf = quadtree.leaves[0]
    assert aleaf == quadtree.find_quads([aleaf.center_point])[0], "Leaf found does not match expected leaf"

    manyleaves = quadtree.leaves[:30]
    quads = quadtree.find_quads([quad.center_point for quad in manyleaves])
    assert np.array([manyleaves[idx] == quads[idx] for idx in range(len(quads))]).all(), "Many quads search failed"
    
if __name__ == "__main__":
    test_quad_on_simple_pf()

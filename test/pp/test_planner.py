
import larp

"""
Author: Josue N Rivera

"""

def test_planner():
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
                                  minimum_length_limit=5)
    
    planner = larp.pp.QuadPlanner(quadtree, 'a*')

    path = planner.find_path((45, 45), (65, 65), reset_memory=True)
    
    assert path is not None, "No path was returned when a path was expected"

test_planner()
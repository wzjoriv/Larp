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
                                  minimum_sector_length=5,
                                  boundaries=np.arange(0.2, 0.8, 0.2))
    quadtree.build()
    routing_graph = larp.graph.RouteGraph(quad_tree=quadtree)
    routing_graph.find_route((45, 45), (60, 65), alg='A*')
    route = routing_graph.find_route((45, 45), (60, 65), alg='Dijkstra')

    route_path = larp.graph.RouteGraph.route_to_lines_collection((45, 45), (60, 65), route, remapped=True)
    
    
if __name__ == "__main__":
    test_quad_on_simple_pf()
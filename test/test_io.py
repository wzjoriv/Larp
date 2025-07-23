import numpy as np
import sys
sys.path.append("../larp")
import larp.io as lpio

"""
Author: Josue N Rivera
"""

def test_load_rgj():

    field = lpio.loadRGeoJSONFile("test/data.rgj")
    field.eval([[0.0, 0.0]])

def test_load_gj():
    field = lpio.loadGeoJSONFile("test/data.geojson")
    field.eval([[0.0, 0.0]])

def test_load_quadtree():
    quadtree = lpio.loadQuadTreeFile('test/data.quad.lp')
    field = quadtree.field
    field.eval([[55.0, 55.0]])

def test_load_occupancy_map():

    grid = np.array([
        [0, 0, 0],
        [1, 1, 0]
    ])
    field = lpio.loadOccupancyMap(grid)
    assert len(field.rgjs) == 2
    assert np.allclose(field.size, np.array([3, 2]))
    assert field.in_bbox((1.5, 1.5)) == True

test_load_occupancy_map()
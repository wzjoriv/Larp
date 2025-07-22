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
        [0, 1],
        [1, 0]
    ])
    field = lpio.loadOccupancyMap(grid, cell_size=1.0)
    assert len(field.rgjs) == 2

test_load_occupancy_map()
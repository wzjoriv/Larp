import numpy as np
import sys
sys.path.append("../larp")
import larp
import larp.io as lpio

"""
Author: Josue N Rivera
"""

def test_load_rgj():

    field = lpio.loadRGeoJSONFile("test/data.rgj")
    field.eval([[0.0, 0.0]])

test_load_rgj()

def test_load_gj():
    field = lpio.loadGeoJSONFile("test/data2.geojson")
    field.eval([[0.0, 0.0]])

def test_load_quadtree():
    quadtree = lpio.loadQuadTreeFile('test/data.quad.larp')
    field = quadtree.field
    field.eval([[55.0, 55.0]])

test_load_quadtree()
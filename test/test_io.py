import os
import tempfile
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

def test_load_gj():
    field = lpio.loadGeoJSONFile("test/data.geojson")
    field.eval([[0.0, 0.0]])

def test_load_quadtree():
    # `test/data.quad.lp` was a committed pickle fixture that was later
    # removed from the repo (a9c7aac) without updating this test, and a
    # pickled QuadTree would go stale the moment RiskField's class layout
    # changes anyway -- so build+save+load a fresh one instead.
    field = lpio.loadRGeoJSONFile("test/data.rgj")
    quadtree = larp.QuadTree(field, minimum_length_limit=min(field.size) / 16.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.quad.lp")
        lpio.saveQuadTree(quadtree, path)
        loaded_quadtree = lpio.loadQuadTreeFile(path)

    field = loaded_quadtree.field
    field.eval([[55.0, 55.0]])


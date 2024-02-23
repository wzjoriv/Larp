import numpy as np
import sys
sys.path.append("../LARP")

import larp

"""
Author: Josue N Rivera
"""

def test_point_rgj():
    point_rgj = {
        'type': "Point",
        'coordinates': [50, 50], 
        'repulsion': [[10, 0], [0, 5]]
    }

    point_rgj = larp.PointRGJ(coordinates = np.array(point_rgj["coordinates"]), repulsion= np.array(point_rgj["repulsion"]))

    x = np.array([[50, 65], [55, 55], [50, 50]])
    out:np.ndarray = point_rgj.eval(x)
    
    assert np.squeeze(out)[2] == 1.0, "Evaluation of point rgj is incorrect for point at origin"
    assert len(np.squeeze(out)) == 3, "Evaluation of point rgj does not return a size equal to the size of the input"

test_point_rgj()

def test_line_string_rgj():
    lines_rgj = {
        'type': "LineString",
        'coordinates': [[62, 53], [62, 60], [65, 65], [60, 60]], 
        'repulsion': [[5, 0], [0, 5]]
    }

    lines_rgj = larp.LineStringRGJ(coordinates = np.array(lines_rgj["coordinates"]), repulsion = np.array(lines_rgj["repulsion"]))

    x = np.array([[62, 53], [65, 65], [60, 60], [62, 62]])

    # Test points in line string pairing
    out:np.ndarray = lines_rgj.points_in_line_pair
    assert np.all(out[0] == lines_rgj.coordinates[:2]),  "Points 1 and 2 not paired correctly as a line"
    assert np.all(out[1] == lines_rgj.coordinates[1:3]), "Points 2 and 3 not paired correctly as a line"
    assert np.all(out[2] == lines_rgj.coordinates[2:]),  "Points 3 and 4 not paired correctly as a line"

    # Test individual line
    out:np.ndarray = lines_rgj.__squared_dist_one_line__(x, line=lines_rgj.points_in_line_pair[0])
    assert np.squeeze(out)[0] == 0.0, "Evaluation of line string rgj is incorrect for point at an origin"

    # Test group of lines
    out:np.ndarray = lines_rgj.eval(x)
    assert len(np.squeeze(out)) == len(x), "Evaluation of line string rgj does not return a size equal to the size of the input"
    assert np.squeeze(out)[0] == 1.0, "Evaluation of line string rgj is incorrect for point at an origin"
    assert np.squeeze(out)[1] == 1.0, "Evaluation of line string rgj is incorrect for point at an origin"
    assert np.squeeze(out)[2] == 1.0, "Evaluation of line string rgj is incorrect for point at an origin"
    assert np.squeeze(out)[3] == 1.0, "Evaluation of line string rgj is incorrect for point at an origin"

test_line_string_rgj()
    
    
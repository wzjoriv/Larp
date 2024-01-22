import numpy as np
import sys
sys.path.append("E:\dev\Projects\LARP")

import larp

def test_point_loi():
    point_loi = {
        'type': "Point",
        'coordinates': [50, 50], 
        'decay': [[10, 0], [0, 5]]
    }

    point_loi = larp.PointLOI(coordinates = np.array(point_loi["coordinates"]), decay = np.array(point_loi["decay"]))

    x = np.array([[50, 65], [55, 55], [50, 50]])
    out:np.ndarray = point_loi.eval(x)
    
    assert np.squeeze(out)[2][()] == 1.0, "Evaluation of point loi is incorrect for point at origin"
    assert len(np.squeeze(out)) == 3, "Evaluation of point loi does not return a size equal to the size of the input"

test_point_loi()

def test_line_string_loi():
    lines_loi = {
        'type': "LineString",
        'coordinates': [[62, 53], [62, 60], [65, 65], [60, 60]], 
        'decay': [[5, 0], [0, 5]]
    }

    lines_loi = larp.LineStringLOI(coordinates = np.array(lines_loi["coordinates"]), decay = np.array(lines_loi["decay"]))

    x = np.array([[62, 53], [65, 65], [60, 60], [62, 62]])
    out:np.ndarray = lines_loi.eval(x)

    assert np.squeeze(out)[0][()] == 1.0, "Evaluation of line string loi is incorrect for point at an origin"
    assert np.squeeze(out)[1][()] == 1.0, "Evaluation of line string loi is incorrect for point at an origin"
    assert np.squeeze(out)[2][()] == 1.0, "Evaluation of line string loi is incorrect for point at an origin"
    assert np.squeeze(out)[3][()] == 1.0, "Evaluation of line string loi is incorrect for point at an origin"
    assert len(np.squeeze(out)) == len(x), "Evaluation of line string loi does not return a size equal to the size of the input"

test_line_string_loi()

    # lines_loi = larp.LineStringLOI(coordinates = np.array(lines_loi["coordinates"]), decay = np.array(lines_loi["decay"]))
    
    
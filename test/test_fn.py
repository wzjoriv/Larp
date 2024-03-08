import numpy as np
import sys

sys.path.append("../larp")
import larp

"""
Author: Josue N Rivera

"""

def test_route_distance():
    dist = larp.route_distance([(0, 0), (1, 1)])

    assert ((dist - np.sqrt(2))**2).sum() < 1e-10, "Distance incorrect for diagonal line"

    dist, joints_dist = larp.route_distance([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)], return_joints=True)

    assert ((dist - 4)**2).sum() < 1e-10, "Distance incorrect for a square"
    assert ((joints_dist - np.array([1, 2, 3, 4]))**2).sum() < 1e-10, "Joint distance incorrect for a square"

def test_interpolate_route():
    points = larp.interpolate_along_route([(0, 0), (1, 1)], step=0.5)

    assert (points[1] != np.array([0.5, 0.5])).all(), "Middle point interpolated using wrong projection"
    assert len(points) == 3, "Number of points not expected"

    points = larp.interpolate_along_route([(0, 0), (1, 1)], step=1.0/np.sqrt(2))

    assert ((points[1] - np.array([0.5, 0.5]))**2).sum() < 1e-10, "Middle point interpolated incorrectly"
    assert len(points) == 3, "Number of points not expected (3 due to rounding error)"

    points, step, n = larp.interpolate_along_route([(0, 0), (1, 1)], n = 3, return_step_n=True)
    assert (points[1] == np.array([0.5, 0.5])).all(), "Middle point interpolated incorrectly with n parameter"
    assert n == 3, "Returned n not expected"
    assert ((step - 1.0/np.sqrt(2))**2).sum() < 1e-10, "Returned step not expected"
    assert len(points) == 3, "Number of points not expected"
    
if __name__ == "__main__":
    test_route_distance()
    test_interpolate_route()
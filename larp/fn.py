
from typing import List, Tuple, Union
import numpy as np
from larp.types import Point

"""
Author: Josue N Rivera
"""

def route_distance(route:Union[np.ndarray, List[Point]], return_joints = False) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Return distance of a path defined by a list of continuous points
    """
    route = np.array(route)

    dist = np.linalg.norm(route[1:, :] - route[:-1, :], axis=1)

    if return_joints:
        cascade = np.cumsum(dist)
        return cascade[-1], cascade
    
    return dist.sum()

def interpolate_along_route(route:Union[List[Point], np.ndarray], step=1e-3, n=0, return_step_n = False) -> Union[np.ndarray, Tuple[np.ndarray, float, int]]:
    """
    Return a set of equally spaced points along a route
    """
    route = np.array(route)
    total_dist, joints_dist = route_distance(route, return_joints=True)
    if n <= 0:
        offset = np.arange(0.0, total_dist, step)
        n = len(offset)
    else:
        offset = np.linspace(0.0, total_dist, n)
        step = total_dist/(n-1)

    lines_idx = np.digitize(offset, joints_dist, right=True)
    relative_offset = offset - np.concatenate([[0.0], joints_dist])[lines_idx]

    line_starts = route[:-1]
    line_ends = route[1:]
    lines_diff = line_ends - line_starts
    uni_vectors = lines_diff/np.linalg.norm(lines_diff, axis=1, keepdims=True)

    points = line_starts[lines_idx] + uni_vectors[lines_idx]*relative_offset.reshape(-1, 1)

    return points if not return_step_n else (points, step, n)
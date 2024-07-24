
from typing import List, Tuple, Union
import numpy as np
from larp.types import Point
from pyproj import CRS, Transformer

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

def project_route(route:Union[List[Point], np.ndarray], from_crs="EPSG:3857", to_crs="EPSG:4326") -> np.ndarray:

    return project_points(points=route, from_crs=from_crs, to_crs=to_crs)

def project_points(points:Union[Point, List[Point], np.ndarray], from_crs="EPSG:3857", to_crs="EPSG:4326") -> np.ndarray:

    from_crs = CRS(from_crs)
    to_crs = CRS(to_crs)

    proj = Transformer.from_crs(crs_from=from_crs, crs_to=to_crs)
    
    points = np.array(points)
    points = np.expand_dims(points, axis=0) if points.ndim < 2 else points

    return np.stack(proj.transform(points[:,0], points[:, 1]), axis=1)

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
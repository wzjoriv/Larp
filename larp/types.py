from typing import Any, List, Tuple, TypedDict, Union
import numpy as np

"""
Author: Josue N Rivera

TODO: Add quads and eclipse
"""

FieldSize = Tuple[int, int]

Point = Union(Tuple[float, float], np.ndarray)

LOIDict = TypedDict('LOIDict', { #super set of GeoJSON geometry
    'type': str,
    'coordinates': Any, 
    'decay': List[List[float]]
})

PointLOIDict = TypedDict('PointLOIDict', {
    'type': str,
    'coordinates': List[float], 
    'decay': List[List[float]]
})

LineStringLOIDict = TypedDict('LineStringLOIDict',{
    'type': str,
    'coordinates': List[float], 
    'decay': List[List[float]]
})
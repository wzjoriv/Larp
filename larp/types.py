from typing import Any, List, Tuple, TypedDict
import numpy as np

"""
Author: Josue N Rivera
"""

FieldSize = Tuple[int, int]

Point = Tuple[float, float]

LOI = TypedDict('LOI', { #super set of GeoJSON geometry
    'type': str,
    'coordinates': Any, 
    'decay': List[List[float]]
})

PointLOI = TypedDict('PointLOI', {
    'type': str,
    'coordinates': List[float], 
    'decay': List[List[float]]
})

LineStringLOI = TypedDict('LineStringLOI',{
    'type': str,
    'coordinates': List[float], 
    'decay': List[List[float]]
})
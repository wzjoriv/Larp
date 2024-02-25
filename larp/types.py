from typing import Any, List, Tuple, TypedDict, Union
import numpy as np

"""
Author: Josue N Rivera

TODO: Add quads and eclipse
"""

FieldSize = Tuple[int, int]

Point = Union[Tuple[float, float], np.ndarray]
RepulsionVectorsAndRef = Tuple[List[int], np.ndarray]

RGJDict = TypedDict('LOIDict', { #super set of GeoJSON geometry
    'type': str,
    'coordinates': Any, 
    'repulsion': List[List[float]]
})

PointRGJDict = TypedDict('PointLOIDict', {
    'type': str,
    'coordinates': List[float], 
    'repulsion': List[List[float]]
})

LineStringRGJDict = TypedDict('LineStringLOIDict',{
    'type': str,
    'coordinates': List[float], 
    'repulsion': List[List[float]]
})
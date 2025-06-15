from typing import Any, Callable, List, Tuple, TypedDict, Union, Literal, TYPE_CHECKING

"""
Author: Josue N Rivera

TODO: Add quads and eclipse
"""

FieldSize = Tuple[int, int]

import numpy as np

ArrayLike = np.ndarray

Point = Union[Tuple[float, float], ArrayLike]
RepulsionVectorsAndRef = Tuple[List[int], ArrayLike]

RoutingAlgorithmStr = Literal['a*', 'dijkstra']

Scaler = Callable[[float], float]

RGJDict = TypedDict('LOIDict', { #super set of GeoJSON geometry
    'type': str,
    'coordinates': Any, 
    'repulsion': List[List[float]]
})

RGeoJSONObject = TypedDict('RGeoJSONObject', { 
    'type': Literal["Feature"],
    'properties': dict, 
    'geometry': RGJDict
})

RGeoJSONCollection = TypedDict('RGeoJSONCollection', { 
    'type': Literal["FeatureCollection"],
    'features': List[RGeoJSONObject]
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
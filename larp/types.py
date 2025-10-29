from typing import Any, Callable, List, Optional, Tuple, TypeAlias, TypedDict, Union, Literal
from numpy import ndarray

"""
Author: Josue N Rivera

TODO: Add quads and eclipse
"""

FieldSize = Tuple[int, int]

FieldConstraints:TypeAlias = Tuple[Optional[ndarray], Optional[ndarray]]


TrajHorizon:TypeAlias = Union[int, float]
Trajectory:TypeAlias = Tuple[ndarray, ndarray]

Point = Union[Tuple[float, float], ndarray]
Path:TypeAlias = Union[ndarray, List[Point]]
RepulsionVectorsAndRef = Tuple[List[int], ndarray]

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
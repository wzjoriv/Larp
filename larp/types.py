from typing import Any, Callable, List, Tuple, TypedDict, Union, Literal
import numpy as np

"""
Author: Josue N Rivera

TODO: Add quads and eclipse
"""

FieldSize = Tuple[int, int]

Point = Union[Tuple[float, float], np.ndarray]
RepulsionVectorsAndRef = Tuple[List[int], np.ndarray]

RoutingAlgorithmStr = Literal['a*', 'dijkstra']

FieldScaleTransform = Callable[[float], float]

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
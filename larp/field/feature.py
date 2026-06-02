from dataclasses import dataclass

from larp.field.geometry.geometry import RGJGeometry


@dataclass
class Feature:
    type:str = 'Feature'
    geometry:dict|RGJGeometry
    properties:dict = {}

@dataclass
class FeatureCollection:
    type:str = 'FeatureCollection'
    features:list[Feature]
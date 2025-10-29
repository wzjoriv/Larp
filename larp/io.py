from os import PathLike
from typing import Union
import numpy as np
from larp.field import PotentialField, RGJGeometry
from larp.quad import QuadTree
from pyproj import CRS, Transformer
import json
import pickle

"""
Author: Josue N Rivera
"""

def saveRGeoJSON(field:PotentialField, file:Union[str, PathLike], return_bbox=False):

    with open(file, "w", encoding='utf-8') as outfile:
        json.dump(field.toRGeoJSON(return_bbox=return_bbox), outfile)

def saveQuadTree(tree:QuadTree, file:Union[str, PathLike]):

    data = tree.toDict()

    with open(file, "wb") as outfile:
        pickle.dump(data, outfile)

def loadRGeoJSON(rgeojson: dict, size_offset = 0.0) -> PotentialField:

    features = rgeojson["features"]
    rgjs = [feature["geometry"] for feature in features]
    properties = [feature["properties"] for feature in features]

    field = PotentialField(rgjs=rgjs, properties=properties, extra_info={key: rgeojson[key] for key in rgeojson if key.lower() not in ["features", "type"]})
    field.size += size_offset*2

    return field

def loadRGeoJSONFile(file: Union[str, PathLike], size_offset = 0.0) -> PotentialField:

    with open(file=file, mode='r', encoding='utf-8') as f:
        rgeojson = json.load(f)

    return loadRGeoJSON(rgeojson, size_offset=size_offset)

def loadQuadTreeFile(file: Union[str, PathLike], size_offset = 0.0, return_field = False) -> QuadTree:

    with open(file=file, mode='rb') as f:
        data:dict = pickle.load(f)

    field = loadRGeoJSON(data.pop('field'), size_offset=size_offset)
    tree = QuadTree(field=field)
    tree.fromDict(data=data)

    if return_field:
        return tree, field

    return tree

def loadGeoJSON(geojson: Union[dict, str], size_offset = 0.0):

    """
    Converts a GeoJSON into an RGeoJSON

    _Note_: All polygons will be converted to line strings.
    """
    if isinstance(geojson, str):
        geojson = json.loads(geojson)

    def __prune_polygons__(geojson:dict):
        if geojson["type"].lower() == "polygon":

            if len(geojson["coordinates"]) == 1:
                geojson["type"] = "LineString"
                geojson["coordinates"] = geojson["coordinates"][0]
            else:
                geojson["type"] = "MultiLineString"

        elif geojson["type"].lower() == "multipolygon":
            geojson["type"] = "MultiLineString"

            coords = []
            for coord in geojson["coordinates"]:
                coords.extend(coord)
            geojson["coordinates"] = coords

        elif geojson["type"].lower() == "geometrycollection":
            for idx in range(len(geojson["geometries"])):
                __prune_polygons__(geojson["geometries"][idx])

    for idx in range(len(geojson["features"])):
        __prune_polygons__(geojson["features"][idx]["geometry"])

    return loadRGeoJSON(geojson, size_offset=size_offset)

def loadGeoJSONFile(file: Union[str, PathLike], size_offset = 0.0):
    
    with open(file=file, mode='r', encoding='utf-8') as f:
        geojson = json.load(f)

    return loadGeoJSON(geojson, size_offset=size_offset)

def projectCoordinates(field: PotentialField, from_crs="EPSG:4326", to_crs="EPSG:3857", recal_size=True):

    from_crs = CRS(from_crs)
    to_crs = CRS(to_crs)

    proj = Transformer.from_crs(crs_from=from_crs, crs_to=to_crs)

    def __prune_coords__(rgj:RGJGeometry):

        if rgj.RGJType.lower() == "geometrycollection":
            for rgj_n in rgj.rgjs:
                __prune_coords__(rgj_n)
        elif rgj.RGJType.lower() in ["point", "ellipse"]:
            rgj.set_coordinates(np.array(proj.transform(rgj.coordinates[0], rgj.coordinates[1])))
        elif rgj.RGJType.lower() in ["linestring", "rectangle", "multipoint", "multiellipse"]:
            rgj.set_coordinates(np.stack(proj.transform(rgj.coordinates[:,0], rgj.coordinates[:, 1]), axis=1))
        elif rgj.RGJType.lower() == "multirectangle":
            rgj.set_coordinates(np.stack(proj.transform(rgj.coordinates[:,:,0], rgj.coordinates[:,:,1]), axis=2))
        elif rgj.RGJType.lower() == "multilinestring":
            new_coords = []
            for coord_idx in range(len(rgj.coordinates)):
                new_coords.append(np.stack(proj.transform(rgj.coordinates[coord_idx][:, 0],\
                                                                     rgj.coordinates[coord_idx][:, 1]), axis=1))
            rgj.set_coordinates(new_coords)

    for rgj in field:
        __prune_coords__(rgj)

    if recal_size:
        field.reload_bbox()
        field.reload_center_point(toggle=True, recal_size=True)

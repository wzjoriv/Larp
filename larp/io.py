from os import PathLike
from typing import Union
import numpy as np
from larp.field import RiskField, RGJGeometry
from larp.quad import QuadTree
from pyproj import CRS, Transformer
import json
import pickle

"""
Author: Josue N Rivera
"""

def saveRGeoJSON(field:RiskField, file:Union[str, PathLike], return_bbox=False):

    with open(file, "w", encoding='utf-8') as outfile:
        json.dump(field.toRGeoJSON(return_bbox=return_bbox), outfile)

def saveQuadTree(tree:QuadTree, file:Union[str, PathLike]):

    data = tree.toDict()

    with open(file, "wb") as outfile:
        pickle.dump(data, outfile)

def loadRGeoJSON(rgeojson: dict, size_offset = 0.0) -> RiskField:

    features = rgeojson["features"]
    rgjs = [feature["geometry"] for feature in features]
    properties = [feature["properties"] for feature in features]

    field = RiskField(rgjs=rgjs, properties=properties, extra_info={key: rgeojson[key] for key in rgeojson if key.lower() not in ["features", "type"]})
    field.size += size_offset*2

    return field

def loadShapely():
    raise NotImplementedError("Not implemented yet. Coming soon.")

def loadRGeoJSONFile(file: Union[str, PathLike], size_offset = 0.0) -> RiskField:

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

        if geojson["type"].lower() == "geometrycollection":
            for idx in range(len(geojson["geometries"])):
                __prune_polygons__(geojson["geometries"][idx])

    return loadRGeoJSON(geojson, size_offset=size_offset)

def loadGeoJSONFile(file: Union[str, PathLike], size_offset = 0.0):
    
    with open(file=file, mode='r', encoding='utf-8') as f:
        geojson = json.load(f)

    return loadGeoJSON(geojson, size_offset=size_offset)

def projectCoordinates(field: RiskField, from_crs="EPSG:4326", to_crs="EPSG:3857", recal_size=True):

    from_crs = CRS(from_crs)
    to_crs = CRS(to_crs)

    proj = Transformer.from_crs(crs_from=from_crs, crs_to=to_crs)

    def __prune_coords__(rgj:RGJGeometry):

        t_type = rgj.RGJType.lower()

        if t_type == "geometrycollection":
            for rgj_n in rgj.rgjs:
                __prune_coords__(rgj_n)
        
        elif t_type in ["point", "ellipse"]:
            rgj.set_coordinates(np.array(proj.transform(rgj.coordinates[0], rgj.coordinates[1])))
        
        elif t_type in ["linestring", "rectangle", "multipoint", "multiellipse"]:
            rgj.set_coordinates(np.stack(proj.transform(rgj.coordinates[:,0], rgj.coordinates[:, 1]), axis=1))
        
        elif t_type == "multirectangle":
            rgj.set_coordinates(np.stack(proj.transform(rgj.coordinates[:,:,0], rgj.coordinates[:,:,1]), axis=2))
        
        elif t_type in ["polygon", "multilinestring"]:
            new_coords = []
            for ring in rgj.coordinates:
                ring_array = np.asarray(ring)
                projected_ring = np.stack(proj.transform(ring_array[:, 0], ring_array[:, 1]), axis=1)
                new_coords.append(projected_ring)
            rgj.set_coordinates(new_coords)

        elif t_type == "multipolygon":
            new_poly_coords = []
            for polygon in rgj.coordinates:
                new_rings = []
                for ring in polygon:
                    ring_array = np.asarray(ring)
                    projected_ring = np.stack(proj.transform(ring_array[:, 0], ring_array[:, 1]), axis=1)
                    new_rings.append(projected_ring)
                new_poly_coords.append(new_rings)
            rgj.set_coordinates(new_poly_coords)

    for rgj in field:
        __prune_coords__(rgj)

    if recal_size:
        field.reload_bbox()
        field.reload_center_point(toggle=True, recal_size=True)

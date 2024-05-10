import numpy as np
from larp.field import PotentialField, RGJGeometry
from larp.quad import QuadTree
from pyproj import CRS, Transformer
import json
import pickle

"""
Author: Josue N Rivera
"""

def saveRGeoJSON(field:PotentialField, file:str, return_bbox=False):

    with open(file, "w") as outfile:
        json.dump(field.toRGeoJSON(return_bbox=return_bbox), outfile)

def saveQuadTree(tree:QuadTree, file:str):

    with open(file, "w") as outfile:
        pickle.dump(tree, outfile)

def fromRGeoJSON(rgeojson: dict, size_offset = 0.0) -> PotentialField:

    features = rgeojson["features"]
    rgjs = [feature["geometry"] for feature in features]
    properties = [feature["properties"] for feature in features]

    field = PotentialField(rgjs=rgjs, properties=properties, extra_info={key: rgeojson[key] for key in rgeojson if key.lower() not in ["features", "type"]})
    field.size += size_offset*2

    return field

def loadRGeoJSONFile(file: str, size_offset = 0.0) -> PotentialField:

    with open(file=file, mode='r') as f:
        rgeojson = json.load(f)

    return fromRGeoJSON(rgeojson, size_offset=size_offset)

def loadQuadTree(file: str) -> QuadTree:

    with open(file=file, mode='r') as f:
        tree = pickle.load(f)

    return tree

def fromGeoJSON(geojson: dict, size_offset = 0.0):

    """
    Converts a GeoJSON into an RGeoJSON

    _Note_: All polygons will be converted to line strings.
    """

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

    return fromRGeoJSON(geojson, size_offset=size_offset)

def loadGeoJSONFile(file: str, size_offset = 0.0):

    with open(file=file) as f:
        geojson = json.load(f)

    return fromGeoJSON(geojson, size_offset=size_offset)

def projectCoordinates(field: PotentialField, from_crs="EPSG:4326", to_crs="EPSG:3857", recalculate=True):

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

    if recalculate:
        field.reload_center_point(toggle=True, recal_size=True)

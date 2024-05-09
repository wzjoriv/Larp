from larp.field import PotentialField
import pyproj as pj
import json

"""
Author: Josue N Rivera
"""

def fromRGeoJSON(rgeojson: dict, size_offset = 0.0) -> PotentialField:

    features = rgeojson["features"]
    rgjs = [feature["geometry"] for feature in features]
    properties = [feature["properties"] for feature in features]

    field = PotentialField(rgjs=rgjs, properties=properties, extra_info={key: rgeojson[key] for key in rgeojson if key.lower() not in ["features", "type"]})
    field.size += size_offset

    return field

def loadRGeoJSONFile(file: str) -> PotentialField:

    with open(file=file, mode='r') as f:
        rgeojson = json.load(f)

    return fromRGeoJSON(rgeojson)

def saveRGeoJSON(field:PotentialField, file:str, return_bbox=False):

    with open("sample.json", "w") as outfile:
        json.dump(field.toRGeoJSON(return_bbox=return_bbox), outfile)

def loadGeoJSONFile(file: str):

    """
    Converts a GeoJSON into an RGeoJSON

    _Note_: All polygons will be converted to line strings.
    """

    with open(file=file) as f:
        geojson = json.load(f)

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

    return fromRGeoJSON(geojson)

def projectCoordinates(field: PotentialField, from_crs="EPSG:4326", to_crs="EPSG:26917"):
    raise NotImplementedError
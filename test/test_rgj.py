import numpy as np
import larp

"""
Author: Josue N Rivera
"""

def test_point_rgj():
    point_rgj = {
        'type': "Point",
        'coordinates': [50, 50], 
        'repulsion': [[10, 0], [0, 5]]
    }

    point_rgj = larp.PointRGJ(coordinates = np.array(point_rgj["coordinates"]), repulsion= np.array(point_rgj["repulsion"]))

    x = np.array([[50, 65], [55, 55], [50, 50]])
    out:np.ndarray = point_rgj.eval(x)
    
    assert np.squeeze(out)[2] == 1.0, "Evaluation of point rgj is incorrect for point at origin"
    assert len(np.squeeze(out)) == 3, "Evaluation of point rgj does not return a size equal to the size of the input"

def test_line_string_rgj():
    lines_rgj = {
        'type': "LineString",
        'coordinates': [[62, 53], [62, 60], [65, 65], [60, 60]], 
        'repulsion': [[5, 0], [0, 5]]
    }

    lines_rgj = larp.LineStringRGJ(coordinates = np.array(lines_rgj["coordinates"]), repulsion = np.array(lines_rgj["repulsion"]))

    x = np.array([[62, 53], [65, 65], [60, 60], [62, 62]])

    # Test points in line string pairing
    out:np.ndarray = lines_rgj.points_in_line_pair
    assert np.all(out[0] == lines_rgj.coordinates[:2]),  "Points 1 and 2 not paired correctly as a line"
    assert np.all(out[1] == lines_rgj.coordinates[1:3]), "Points 2 and 3 not paired correctly as a line"
    assert np.all(out[2] == lines_rgj.coordinates[2:]),  "Points 3 and 4 not paired correctly as a line"

    # Test group of lines
    out:np.ndarray = lines_rgj.eval(x)
    assert len(np.squeeze(out)) == len(x), "Evaluation of line string rgj does not return a size equal to the size of the input"
    assert np.squeeze(out)[0] == 1.0, "Evaluation of line string rgj is incorrect for point at an origin"
    assert np.squeeze(out)[1] == 1.0, "Evaluation of line string rgj is incorrect for point at an origin"
    assert np.squeeze(out)[2] == 1.0, "Evaluation of line string rgj is incorrect for point at an origin"
    assert np.squeeze(out)[3] == 1.0, "Evaluation of line string rgj is incorrect for point at an origin"

# NOTE: Rectangle/Ellipse/MultiRectangle/MultiEllipse are not part of the
# GeoJSON-standard geometry set carried into the new field architecture
# (Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon,
# GeometryCollection only), so their dedicated tests were removed rather
# than ported. Polygon covers the rectangle case; see test_polygon_rgj.

def test_multi_point_rgj():
    
    rgj = larp.MultiPointRGJ(coordinates=[(1, 1), (0, 0), (0.5, 0.5)], repulsion=np.eye(2))

    vectors = rgj.repulsion_vector([(0.15, 0.15), (0.85, 0.85), (0.45, 0.45), (0.65, 0.65)])

    assert ((vectors[0] - np.array([-0.15, -0.15]))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[1] - np.array([ 0.15,  0.15]))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[2] - np.array([ 0.05,  0.05]))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[3] - np.array([-0.15, -0.15]))**2).sum() < 1e-5, "Unexpected repulsion vector"

def test_multi_line_string_rgj():
    
    rgj = larp.MultiLineStringRGJ(coordinates=[[(0, 1), (0, 0)], [(1, 1), (1, 0), (2, 0)]], repulsion=np.eye(2))

    vectors = rgj.repulsion_vector([(0.15, 0.15), (0.85, 0.15), (0.45, 0.45), (0.55, 0.45)])

    assert ((vectors[0] - np.array([ 0.15, 0.0]))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[1] - np.array([-0.15, 0.0]))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[2] - np.array([ 0.45, 0.0]))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[3] - np.array([-0.45, 0.0]))**2).sum() < 1e-5, "Unexpected repulsion vector"

def test_polygon_rgj():

    # A square (the GeoJSON-standard stand-in for the old RectangleRGJ)
    rgj = larp.PolygonRGJ(coordinates=[[(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)]], repulsion=np.eye(2))

    vectors = rgj.repulsion_vector([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (-2, -2)])

    assert ((vectors[0] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[1] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[2] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[3] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[4] - np.array([-1.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"

def test_geometry_collection_rgj():

    rgj = larp.GeometryCollectionRGJ(geometries=[
         {
            "type": "Point",
            "coordinates": [50, 50],
            "repulsion": [[36, 0], [0, 25]]
        },
        {
            "type": "Point",
            "coordinates": [60, 55],
            "repulsion": [[36, 0], [0, 36]]
        },
        {
            "type": "Point",
            "coordinates": [45, 48],
            "repulsion": [[64, 0], [0, 100]]
        },
        {
            "type": "LineString",
            "coordinates": [[10, 10], [80, 10], [80, 80], [10, 80], [10, 30]],
            "repulsion": [[9, 0], [0, 9]]
        }
    ])

    # (80.0, 10.0) and (70.0, 80.0) lie exactly on the wall LineString;
    # (40, 8) is 2 units below it. All governed by the wall, not the
    # (now-removed) Rectangle/Ellipse obstacles that used to sit elsewhere.
    vectors = rgj.repulsion_vector([(80.0, 10.0), (70.0, 80.0), (40, 8)])

    assert ((vectors[0] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[1] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[2] - np.array([0.0,-2.0]))**2).sum() < 1e-5, "Unexpected repulsion vector"

    values = rgj.eval([(80.0, 10.0), (70.0, 80.0), (40, 8)])

    assert ((values[0] - 1)**2).sum() < 1e-5, "Unexpected risk field evaluation"
    assert ((values[1] - 1)**2).sum() < 1e-5, "Unexpected risk field evaluation"
    assert ((values[2] - np.exp(-4/9))**2).sum() < 1e-5, "Unexpected risk field evaluation"

    grads = rgj.gradient([(80.0, 10.0), (70.0, 80.0), (40, 8)])
    grad_line = rgj.rgjs[3].gradient(np.array([(40, 8)]))[0]

    assert ((grads[0] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected gradient vector"
    assert ((grads[1] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected gradient vector"
    assert ((grads[2] - grad_line)**2).sum() < 1e-5, "Unexpected gradient vector"
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

def test_ellipse_rgj():
    
    rgj = larp.EllipseRGJ(coordinates=np.array([0, 0]), shape=np.eye(2), repulsion=np.eye(2))

    vectors = rgj.repulsion_vector([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])

    assert ((vectors[0] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[1] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[2] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[3] - np.array([(np.sqrt(2)-1)/np.sqrt(2)]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    
    rgj = larp.EllipseRGJ(coordinates=np.array([0, 0]), shape=np.eye(2)*2, repulsion=np.eye(2))
    vectors = rgj.repulsion_vector([(1.0, 1.0), (2.0, 2.0)])

    assert ((vectors[0] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[1] - np.array([2*(np.sqrt(2)-1)/np.sqrt(2)]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"

def test_rect_rgj():
    
    rgj = larp.RectangleRGJ(coordinates=np.array([[1]*2, [-1]*2]), repulsion=np.eye(2))

    vectors = rgj.repulsion_vector([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (-2, -2)])

    assert ((vectors[0] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[1] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[2] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[3] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[4] - np.array([-1.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"

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

def test_multi_rect_rgj():
    
    rgj = larp.MultiRectangleRGJ(coordinates=[[(0, 1), (-1, 0)], [(1, 1), (2, 0)]], repulsion=np.eye(2))

    vectors = rgj.repulsion_vector([(0.15, 0.15), (0.85, 0.15), (0.45, 0.45), (0.55, 0.45)])

    assert ((vectors[0] - np.array([ 0.15, 0.0]))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[1] - np.array([-0.15, 0.0]))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[2] - np.array([ 0.45, 0.0]))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[3] - np.array([-0.45, 0.0]))**2).sum() < 1e-5, "Unexpected repulsion vector"

def test_multi_ellipse_rgj():
    
    rgj = larp.MultiEllipseRGJ(coordinates=[(0, 0), (1, 1)], shape=np.array([np.eye(2)*0.25]*2), repulsion=np.eye(2))

    vectors = rgj.repulsion_vector([(0.15, 0.15), (0.85, 0.85), (0.45, 0.45), (0.55, 0.55)])

    assert ((vectors[0] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[1] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[2] - (1-1/np.sqrt(6.48))*np.array([ 0.45, 0.45]))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[3] - (1-1/np.sqrt(6.48))*np.array([-0.45,-0.45]))**2).sum() < 1e-5, "Unexpected repulsion vector"

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
        },
        {
            "type": "Rectangle",
            "coordinates": [[68, 22], [85, 0]], 
            "repulsion": [[4, 0], [0, 4]]
        },
        {
            "type": "Ellipse",
            "coordinates": [10, 80], 
            "repulsion": [[9, 0], [0, 9]],
            "shape": [[4, 0], [0, 4]]
        }
    ])

    vectors = rgj.repulsion_vector([(80.0, 10.0), (11, 81), (70.0, 80.0), (40, 8)])

    assert ((vectors[0] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[1] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[2] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected repulsion vector"
    assert ((vectors[3] - np.array([0.0,-2.0]))**2).sum() < 1e-5, "Unexpected repulsion vector"

    values = rgj.eval([(80.0, 10.0), (11, 81), (70.0, 80.0), (40, 8)])

    assert ((values[0] - 1)**2).sum() < 1e-5, "Unexpected potential field evaluation"
    assert ((values[1] - 1)**2).sum() < 1e-5, "Unexpected potential field evaluation"
    assert ((values[2] - 1)**2).sum() < 1e-5, "Unexpected potential field evaluation"
    assert ((values[3] - np.exp(-4/9))**2).sum() < 1e-5, "Unexpected potential field evaluation"

    grads = rgj.gradient([(80.0, 10.0), (11, 81), (70.0, 80.0), (40, 8)])
    grad_line = rgj.rgjs[3].gradient(np.array([(40, 8)]))[0]

    assert ((grads[0] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected gradient vector"
    assert ((grads[1] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected gradient vector"
    assert ((grads[2] - np.array([0.0]*2))**2).sum() < 1e-5, "Unexpected gradient vector"
    assert ((grads[3] - grad_line)**2).sum() < 1e-5, "Unexpected gradient vector"
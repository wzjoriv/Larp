import json
import numpy as np
import sys
sys.path.append("../larp")
import larp

"""
Author: Josue N Rivera
"""

def test_eval():

    rgjs = [
        {
            "type": "Point",
            "coordinates": [50, 50], 
            "repulsion": [[100, 0], [0, 25]]
        },
        {
            "type": "Point",
            "coordinates": [60, 55], 
            "repulsion": [[144, 0], [0, 144]]
        },
        {
            "type": "Point",
            "coordinates": [55, 48], 
            "repulsion": [[64, 0], [0, 100]]
        },
        {
            "type": "LineString",
            "coordinates": [[62, 53], [62, 60], [65, 65], [60, 60]], 
            "repulsion": [[25, 0], [0, 25]]
        }
    ]
    
    field = larp.PotentialField(size=(100, 100), rgjs=rgjs)
    
    x = np.array([[50, 65], [70, 60], [60, 60], [63, 63], [50, 50], [65, 70]])
    
    out = field.eval(x)
    assert len(np.squeeze(out)) == len(x),   "Evaluation of line string rgj does not return a size equal to the size of the input"
    assert np.squeeze(out)[0] != 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[1] != 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[2] == 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[3] == 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[4] == 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[5] == np.exp(-1), "Evaluation of line string rgj is incorrect for a point one magnitude away"

def test_area_estimation():
    rgjs = [
        {
            "type": "Point",
            "coordinates": [50, 50], 
            "repulsion": [[1, 0], [0, 1]]
        }
    ]

    field = larp.PotentialField(size=(100, 100), rgjs=rgjs)
    area = field.estimate_route_area([(49, 50), (51, 50)], step=0.0001)
    assert ((area - 1.49364)**2).sum() < 1e-5, "Area estimation off"

    area = field.estimate_route_area([(49, 50), (51, 50)], step=0.0001, scale_transform=lambda x: 1/(1.0 - x + 0.000001))

def test_gradient():
    rgjs = [
        {
            "type": "Point",
            "coordinates": [50, 50], 
            "repulsion": [[1, 0], [0, 1]]
        },
        {
            "type": "LineString",
            "coordinates": [[10, 10], [10, 20], [20, 20], [20, 10]], 
            "repulsion": [[2, 0], [0, 2]]
        },
        {
            "type": "Rectangle",
            "coordinates": [[30, 30], [25, 25]], 
            "repulsion": [[1, 0], [0, 1]]
        },
        {
            "type": "Ellipse",
            "coordinates": [80, 80], 
            "repulsion": [[4, 0], [0, 4]],
            "shape": [[2, 0], [0, 2]]
        }
    ]

    field = larp.PotentialField(size=(100, 100), rgjs=rgjs)
    grad = field.gradient([(49, 50), (51, 50), (51, 51)])

    assert ((grad[0] - np.array([2*np.exp(-1), 0]))**2).sum() < 1e-5, "Unexpected gradient"
    assert ((grad[1] - np.array([-2*np.exp(-1), 0]))**2).sum() < 1e-5, "Unexpected gradient"
    assert ((grad[2] - np.array([-2*np.exp(-2), -2*np.exp(-2)]))**2).sum() < 1e-5, "Unexpected gradient"

    
    grad = field.gradient([(11, 10), (20, 11), (32, 31), (81, 82)])

def test_bbox():
    rgjs = [
        {
            "type": "Point",
            "coordinates": [50, 50], 
            "repulsion": [[1, 0], [0, 1]]
        },
        {
            "type": "LineString",
            "coordinates": [[10, 10], [10, 20], [20, 20], [20, 10]], 
            "repulsion": [[2, 0], [0, 2]]
        },
        {
            "type": "Rectangle",
            "coordinates": [[30, 30], [25, 25]], 
            "repulsion": [[1, 0], [0, 1]]
        },
        {
            "type": "Ellipse",
            "coordinates": [80, 80], 
            "repulsion": [[4, 0], [0, 4]],
            "shape": [[2, 0], [0, 2]]
        }
    ]

    field = larp.PotentialField(size=(100, 100), rgjs=rgjs)

    assert field.rgjs[1].in_bbox([15, 15]),   "Error determining bbox for linestring"
    assert field.find_bbox([81, 80])[0] == 3, "Error finding bbox for ellipse"
    assert field.find_bbox([15, 15])[0] == 1, "Error finding bbox for linestring"

test_eval()
test_area_estimation()
test_gradient()
test_bbox()
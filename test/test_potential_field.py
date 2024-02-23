import numpy as np
import matplotlib.pyplot as plt
import json
import sys
sys.path.append("../LARP")

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
    
    field = larp.PotentialField(size=(100, 100),
                           rgjs=rgjs)
    
    x = np.array([[50, 65], [70, 60], [60, 60], [63, 63], [50, 50], [65, 70]])
    
    out = field.eval(x)
    assert len(np.squeeze(out)) == len(x),   "Evaluation of line string rgj does not return a size equal to the size of the input"
    assert np.squeeze(out)[0] != 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[1] != 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[2] == 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[3] == 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[4] == 1.0,        "Evaluation of line string rgj is incorrect for point at a high-energy state"
    assert np.squeeze(out)[5] == np.exp(-1), "Evaluation of line string rgj is incorrect for a point one magnitude away"

test_eval()
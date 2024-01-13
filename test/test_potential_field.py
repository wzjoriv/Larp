import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("E:\dev\Projects\LARP")

import larp

"""
Author: Josue N Rivera
"""

if __name__ == "__main__":

    lois = [
        {
            'type': "Point",
            'coordinates': [50, 50], 
            'decay': [[10, 0], [0, 10]]
        },
        {
            'type': "Point",
            'coordinates': [60, 55], 
            'decay': [[12, 0], [0, 12]]
        },
        {
            'type': "Point",
            'coordinates': [55, 48], 
            'decay': [[8, 0], [0, 10]]
        },
        {
            'type': "LineString",
            'coordinates': [[62, 53], [62, 60], [65, 65], [60, 60]], 
            'decay': [[5, 0], [0, 5]]
        }
    ]
    
    field = larp.PotentialField(size=(100, 100),
                           lois=lois)
    
    x2 = np.array([[50, 65]])
    
    print(field.eval(x2))
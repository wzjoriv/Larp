from typing import List

import numpy as np

from larp.types import LOI, FieldSize

"""
Author: Josue N Rivera
"""

class PotentialField():
    """
    Potential field given a subset of LOIs
    """

    def __init__(self, lois:List[LOI]):
        self.lois = lois

        for loi in self.lois:
            loi["coordinates"] = np.array(loi["coordinates"])
            loi["decay"] = np.array(loi["decay"])

        def LineStringEval(x, coordinates, decay, type):
            # TODO: Evalline string
            return 0

        self.funs = {
            "Point": lambda x, coordinates, decay, type: np.exp(-(x - coordinates)*decay*(x - coordinates)),
            "LineString": LineStringEval
        }

    def addLOI(self, loi:LOI) -> None:
        loi["coordinates"] = np.array(loi["coordinates"])
        loi["decay"] = np.array(loi["decay"])

        self.lois.append(loi)

    def evaluate(self, points) -> np.ndarray:
        points = np.array(points)

        return np.sum([self.funs[loi["type"]](points, **loi) for loi in self.lois])
        
    
if __name__ == "__main__":

    pass
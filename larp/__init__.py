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

    def __init__(self, lois:List[LOI], size: FieldSize):
        self.lois = lois
        self.field_size = size

    def addLOI(self, loi:LOI) -> None:
        self.lois.append(loi)

    def evaluate(self, points) -> np.ndarray:

        return np.array(0.0)
        
    
if __name__ == "__main__":

    pass
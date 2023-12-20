from typing import Callable
import numpy as np

from larp.types import FieldSize, PotentialField

"""
Author: Josue N Rivera, Pruthvi Patel
Date: 9/29/2023
Generate the quadtree and graph from the potential field
"""

class QuadNode():
    pass


def PFtoQuads(field: PotentialField,
              size: FieldSize,
              dist_to_meters: float,
              bins:np.ndarray = np.arange(0.1, 1, 0.2)):
    pass


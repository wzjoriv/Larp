from typing import List, Union
import numpy as np
from larp import PotentialField

from larp.types import Point

"""
Author: Josue N Rivera, Pruthvi Patel
Date: 9/29/2023
Generate the quadtree and graph from the potential field
"""

class QuadNode():
    pass


def PFtoQuads(field: PotentialField,
              center_point: Point,
              search_radius: float,
              minimum_sector_length:float = 10.0,
              bins:Union[np.ndarray, List[float]] = np.arange(0.1, 1, 0.2)) -> QuadNode:
    
    loc_tr = np.array(center_point) - np.array([search_radius, search_radius]) # tr = top right
    loc_bl = np.array(center_point) + np.array([search_radius, search_radius]) # bl = bottom left


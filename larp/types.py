from typing import Callable, Tuple
import numpy as np
from larp.quad import QuadNode

"""
Author: Josue N Rivera
"""

PotentialFieldType = Callable[[np.ndarray], QuadNode]
FieldSize = Tuple[int, int]
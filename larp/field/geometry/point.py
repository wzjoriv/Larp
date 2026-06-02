from __future__ import annotations
from abc import ABC
from typing import Optional

import numpy as np
from larp.types import Point

from larp.field.geometry.geometry import RGJGeometry

"""
Author: Josue N Rivera

x are assumed to be a list of point coordinates in euclidean space

"""


__all__ = ["PointRGJ"]


class PointRGJ(RGJGeometry):
    RGJType = "Point"

    def __init__(self, coordinates: np.ndarray | Point, repulsion:np.ndarray | None = None, **kwargs) -> None:
        super().__init__(coordinates=coordinates, repulsion=repulsion, **kwargs)

    def repulsion_vector(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return x - self.coordinates
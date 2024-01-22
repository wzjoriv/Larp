from typing import List, Optional, Union

import numpy as np
from larp.types import LOIDict, FieldSize, Point

"""
Author: Josue N Rivera

coordinates are assumed to be a list of coordinates
"""

class LOI():

    LOIType = None

    def __init__(self, coordinates:np.ndarray, decay:np.ndarray) -> None:
        self.coordinates = np.array(coordinates)
        self.decay = np.array(decay)

    def get_center_point(self):
        return self.coordinates if len(self.coordinates.shape) <= 1 else np.reshape(self.coordinates, (-1, 2)).mean(0)

    def eval(self, x:np.ndarray):
        raise NotImplementedError

    def squared_dist(self, x:np.ndarray):
        raise NotImplementedError

class PointLOI(LOI):
    LOIType = "Point"

    def __init__(self, coordinates: np.ndarray, decay: np.ndarray) -> None:
        super().__init__(coordinates, decay)
    
    def eval(self, x: np.ndarray) -> np.ndarray:

        return np.exp(-(((x - self.coordinates)@self.decay.T)*(x - self.coordinates)).sum(1, keepdims=True))
    
    def squared_dist(self, x: np.ndarray) -> np.ndarray:
        return np.power(x - self.coordinates, 2).sum(1)

class LineStringLOI(LOI):
    LOIType = "LineString"

    def __init__(self, coordinates: np.ndarray, decay: np.ndarray) -> None:
        super().__init__(coordinates, decay)
    
    def __distance_matrix_points__(self, x:np.ndarray, y:Optional[np.ndarray]=None, p:int=2) -> np.ndarray:
        y = self.coordinates if y is None else y

        n = x.shape[0]
        m = y.shape[0]

        x = np.tile(x, (m, 1, 1)).transpose(1, 0, 2)
        y = np.tile(y, (n, 1, 1))

        dist = np.linalg.norm(x - y, ord=p, axis=2)

        return dist

    def __squared_dist_one_line__(self, x: np.ndarray, line: np.ndarray) -> np.ndarray:

        x1x2diff = line[1:2] - line[0:1]
        xx1diff = line[1:2] - line[0:1]

        x12dotxx1 = (x1x2diff*xx1diff).sum(2, keepdims=True)
        x12dotx12 = (x1x2diff*x1x2diff).sum(2, keepdims=True)

        g = line[0] + np.clip(x12dotxx1/x12dotx12, 0.0, 1.0)*(x1x2diff)

        return np.power(x - g, 2).sum(2)

    def __distance_matrix__(self, x, y=None, p=2):
        y = self.coordinates if y is None else y
        dist = 0

        idxs = self.__distance_matrix_points__(x, y).argmin(1)

        return dist

    def eval(self, x: np.ndarray):

        line_idx = self.__distance_matrix__(x, self.coordinates).argmin(1) # get closest line
        line = self.coordinates[line_idx]

        g = line[0] + np.clip(np.dot(line[1] - line[0], x - line[0])/ np.dot(line[1] - line[0], line[1] - line[0]), 0.0, 1.0)*(line[1] - line[0])

        return np.exp(-(x - g)*self.decay*(x - g))
    
    def squared_dist(self, x: np.ndarray) -> np.ndarray:

        return self.__distance_matrix__(x, self.coordinates).min(1)
    

class PotentialField():
    """
    Potential field given a subset of LOIs
    """

    def __init__(self, size:Union[FieldSize, float], center_point: Optional[Point] = None, lois:Optional[List[LOIDict]] = None):
        self.lois:List[LOI] = []
        self.__reload_center = None
        self.size = np.array([size, size]) if np.isscalar(float)  else np.array(size)

        for loi in lois:
            self.addLOI(loi)

        if center_point is None:
            self.__reload_center = True # wether to recalculate center point if new LOI are added
            self.center_point = self.__calculate_center_point()
        else:
            self.__reload_center = False
            self.center_point = center_point

    def __calculate_center_point(self):
        return np.array([loi.get_center_point() for loi in self.lois]).mean(0)

    def addLOI(self, loi:LOIDict) -> None:
        LOIclass:LOI = globals()[loi["type"]+"LOI"](coordinates = np.array(loi["coordinates"]), decay = np.array(loi["decay"]))

        if LOIclass.LOIType != loi["type"]:
            raise RuntimeError("LOI type does not match")
        
        self.lois.append(LOIclass)

        if self.__reload_center:
            self.center_point = self.__calculate_center_point()

    def eval(self, points: Union[np.ndarray, List[Point]]) -> np.ndarray:
        points = np.array(points)

        return np.max(np.concatenate([loi.eval(points) for loi in self.lois], 0), 1)
    
    def squared_dist(self, points:Union[np.ndarray, List[Point]]) -> np.ndarray:
        points = np.array(points)

        return np.min(np.concatenate([loi.squared_dist(points) for loi in self.lois], 1), 1)
    
    def to_display(self, resolution:int = 100, margin:float = 0.0, center_point:Optional[Point] = None):

        if center_point is not None:
            self.center_point = center_point

        n2 = self.size/2.0

        loc_tl = np.array(self.center_point) + np.array([-n2[0]-margin, n2[1]+margin])
        loc_br = np.array(self.center_point) + np.array([n2[0]+margin, -n2[1]-margin])

        xaxis = np.linspace(loc_tl[0], loc_br[0], resolution)
        yaxis = np.linspace(loc_tl[1], loc_br[1], resolution)

        xgrid, ygrid = np.meshgrid(xaxis, yaxis)
        points = np.vstack([xgrid.ravel(), ygrid.ravel()])

        return self.eval(points).reshape((resolution, resolution))
        
from typing import List, Union

import numpy as np

from larp.types import LOIDict, FieldSize

"""
Author: Josue N Rivera

coordinates are assumed to be a list of coordinates
"""

class LOI():

    LOIType = None

    def __init__(self, coordinates:np.ndarray, decay:np.ndarray) -> None:
        self.coordinates = coordinates
        self.decay = decay

    def eval(self, x:np.ndarray):
        raise NotImplementedError

    def squared_dist(self, x:np.ndarray):
        raise NotImplementedError

class PointLOI(LOI):
    LOIType = "Point"

    def __init__(self, coordinates: np.ndarray, decay: np.ndarray) -> None:
        super().__init__(coordinates, decay)
    
    def eval(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-(x - self.coordinates)*self.decay*(x - self.coordinates))
    
    def squared_dist(self, x: np.ndarray) -> np.ndarray:
        return np.power(x - self.coordinates, 2).sum(2)

class LineStringLOI(LOI):
    LOIType = "LineString"

    def __init__(self, coordinates: np.ndarray, decay: np.ndarray) -> None:
        super().__init__(coordinates, decay)

    def __squared_dist_one_line__(self, x: np.ndarray, line: np.ndarray) -> np.ndarray:

        g = line[0] + np.clip(np.dot(line[1] - line[0], x - line[0])/ np.dot(line[1] - line[0], line[1] - line[0]), 0.0, 1.0)*(line[1] - line[0])
        return np.power(x - g, 2).sum(2)

    def __distance_matrix__(self, x, y=None, p=2):
        y = x if y is None else y

        n = x.shape[0]
        m = y.shape[0]
        d = x.shape[1]

        x = np.tile(x[:, np.newaxis, :], (1, m, 1))
        y = np.tile(y[np.newaxis, :, :], (n, 1, 1))

        dist = self.__squared_dist_one_line__(x, y)

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

    def __init__(self, lois:List[LOIDict], size:FieldSize):
        self.lois:List[LOI] = []
        self.size = size

        for loi in lois:
            self.addLOI(loi)


    def addLOI(self, loi:LOIDict) -> None:
        LOIclass:LOI = globals()[loi[type]+"LOI"](coordinates = loi["coordinates"], decay = loi["decay"])

        if LOIclass.LOIType != loi[type]:
            raise RuntimeError("LOI type does not match")
        
        self.lois.append(LOIclass)

    def eval(self, points) -> np.ndarray:
        points = np.array(points)

        return np.max(np.concatenate([loi.eval(points) for loi in self.lois], 1), 1)
    
    def squared_dist(self, x: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        points = np.array(points)

        return np.min(np.concatenate([loi.squared_dist(points) for loi in self.lois], 1), 1)
        
    
if __name__ == "__main__":
    #TODO: test
    pass
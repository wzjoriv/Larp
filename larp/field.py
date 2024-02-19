from multiprocessing import Pool
from typing import List, Optional, Union

import numpy as np
from larp.types import RGJDict, FieldSize, Point

"""
Author: Josue N Rivera

coordinates are assumed to be a list of coordinates
"""

def __distance_matrix_points__(self, x:np.ndarray, y:Optional[np.ndarray]=None, p:int=2) -> np.ndarray:
    y = x if y is None else y

    n = x.shape[0]
    m = y.shape[0]

    x = np.tile(x, (m, 1, 1)).transpose(1, 0, 2)
    y = np.tile(y, (n, 1, 1))

    dist = np.linalg.norm(x - y, ord=p, axis=2)

    return dist

class RGJGeometry():

    RGJType = None

    def __init__(self, coordinates:Union[List[Point], Point], repulsion:np.ndarray, **kwargs) -> None:
        self.coordinates = np.array(coordinates)
        self.repulsion = np.array(repulsion)
        self.inv_repulsion = np.linalg.inv(self.repulsion)

    def get_center_point(self) -> np.ndarray:
        return self.coordinates if len(self.coordinates.shape) <= 1 else np.reshape(self.coordinates, (-1, 2)).mean(0)

    def squared_dist(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def eval(self, x:np.ndarray):
        return np.exp(-self.squared_dist(x).squeeze())

class PointRGJ(RGJGeometry):
    RGJType = "Point"

    def __init__(self, coordinates: np.ndarray, repulsion: np.ndarray, **kwargs) -> None:
        super().__init__(coordinates, repulsion)
    
    def squared_dist(self, x: np.ndarray) -> np.ndarray:

        x_d_xh = x - self.coordinates
        
        return ((x_d_xh@self.inv_repulsion.T)*x_d_xh).sum(1, keepdims=True)

class LineStringRGJ(RGJGeometry):
    RGJType = "LineString"

    def __init__(self, coordinates: np.ndarray, repulsion: np.ndarray, **kwargs) -> None:
        super().__init__(coordinates, repulsion)
        self.lines_n = len(coordinates)
        self.points_in_line_pair = np.stack([self.coordinates[:-1], self.coordinates[1:]], axis=1)

    def __squared_dist_one_line__(self, x: np.ndarray, line: np.ndarray) -> np.ndarray:

        x2_d_x1 = line[1:2] - line[0:1]
        x_d_x1 = x - line[0:1]

        x12dotxx1 = (x2_d_x1*x_d_x1).sum(1, keepdims=True)
        x12dotx12 = (x2_d_x1*x2_d_x1).sum(1, keepdims=True)

        g = line[0] + np.clip(x12dotxx1/x12dotx12, 0.0, 1.0)*(x2_d_x1)
        x_d_g = x - g

        return ((x_d_g@self.inv_repulsion.T)*x_d_g).sum(1, keepdims=True)
    
    def squared_dist(self, x: np.ndarray) -> np.ndarray:
        
        if self.lines_n > 20:
            p = Pool(5)
            dist:np.ndarray = p.map(lambda line: self.__squared_dist_one_line__(x=x, line=line), self.points_in_line_pair)
        else:
            dist:np.ndarray = [self.__squared_dist_one_line__(x=x, line=line) for line in self.points_in_line_pair]

        return np.concatenate(dist, axis=1).min(1)
    
class RectangleRGJ(RGJGeometry):
    RGJType = "Rectangle"

    def __init__(self, coordinates: np.ndarray, repulsion: np.ndarray, **kwargs) -> None:
        super().__init__(coordinates, repulsion)
        self.x1_abs_x2 = np.abs(self.coordinates[0] - self.coordinates[1])

    def __new_vector__(self, x: np.ndarray) -> np.ndarray:
        
        return 0.5*(np.abs(x-self.coordinates[0]) + np.abs(x-self.coordinates[1]) - self.x1_abs_x2)
    
    def squared_dist(self, x: np.ndarray) -> np.ndarray:

        nvector = self.__new_vector__(x)
        return ((nvector@self.inv_repulsion.T)*nvector).sum(1, keepdims=True)
    
class EllipseRGJ(RGJGeometry):
    RGJType = "Ellipse"
    DEN_ERROR_BUFFER = 1e-6

    def __init__(self, coordinates: np.ndarray, repulsion: np.ndarray, shape: np.ndarray, **kwargs) -> None:
        super().__init__(coordinates, repulsion)
        self.shape = shape
        self.inv_shape = np.linalg.inv(self.shape)

    def __new_vector__(self, x: np.ndarray) -> np.ndarray:

        x_d_xh = x - self.coordinates
        Binvx = x_d_xh@self.inv_shape.T

        den = np.sqrt((Binvx*Binvx).sum(1, keepdims=True))
        den = np.maximum(den, self.DEN_ERROR_BUFFER)

        return np.maximum(1 - 1/den, 0)*x_d_xh
    
    def squared_dist(self, x: np.ndarray) -> np.ndarray:

        nvector = self.__new_vector__(x)
        return ((nvector@self.inv_repulsion.T)*nvector).sum(1, keepdims=True)
    

class PotentialField():
    """
    Potential field given a subset of RGJs
    """

    def __init__(self, size:Union[FieldSize, float], center_point: Optional[Point] = None, rgjs:Optional[List[RGJDict]] = None):
        self.rgjs:List[RGJGeometry] = []
        self.__reload_center = None
        self.size = np.array([size, size]) if np.isscalar(float) else np.array(size)

        for rgj in rgjs:
            self.addRGJ(rgj)

        if center_point is None:
            self.__reload_center = True # whether to recalculate center point if new RGJ are added
            self.center_point = self.__calculate_center_point__()
        else:
            self.__reload_center = False
            self.center_point = center_point

    def __calculate_center_point__(self):
        return np.array([rgj.get_center_point() for rgj in self.rgjs]).mean(0)

    def addRGJ(self, rgj_dict:RGJDict) -> None:
        rgj:RGJGeometry = globals()[rgj_dict["type"]+"RGJ"](**rgj_dict)

        if rgj.RGJType != rgj_dict["type"]:
            raise RuntimeError("RGJ type does not match")
        
        self.rgjs.append(rgj)

        if self.__reload_center:
            self.center_point = self.__calculate_center_point__()

    def delRGJ(self, idx:int) -> None:

        del self.rgjs[idx]

    def eval(self, points: Union[np.ndarray, List[Point]], filted_idx:Optional[List[int]] = None) -> np.ndarray:
        points = np.array(points)
        rgjs = self.rgjs[filted_idx] if not (filted_idx is None or len(filted_idx) == 0) else self.rgjs

        return np.max(np.stack([rgj.eval(points) for rgj in rgjs], axis=1), axis=1)
    
    def squared_dist(self, points:Union[np.ndarray, List[Point]], filted_idx:Optional[List[int]] = None) -> np.ndarray:
        points = np.array(points)

        return np.min(self.squared_dist_list(points=points, filted_idx=filted_idx), axis=1)
    
    def squared_dist_list(self, points:Union[np.ndarray, List[Point]], filted_idx:Optional[List[int]] = None) -> np.ndarray:
        points = np.array(points)
        rgjs = self.rgjs[filted_idx] if not (filted_idx is None or len(filted_idx) == 0) else self.rgjs

        return np.concatenate([rgj.squared_dist(points) for rgj in rgjs], axis=1)
    
    def to_image(self, resolution:int = 200, margin:float = 0.0, center_point:Optional[Point] = None) -> np.ndarray:

        if center_point is not None:
            self.center_point = np.array(center_point)

        n2 = self.size/2.0

        loc_tl = np.array(self.center_point) + np.array([-n2[0]-margin, n2[1]+margin])
        loc_br = np.array(self.center_point) + np.array([n2[0]+margin, -n2[1]-margin])


        y_resolution = int(resolution*abs(loc_tl[1] - loc_br[1])/abs(loc_br[0] - loc_tl[0]))
        xaxis = np.linspace(loc_tl[0], loc_br[0], resolution)
        yaxis = np.linspace(loc_tl[1], loc_br[1], y_resolution)

        xgrid, ygrid = np.meshgrid(xaxis, yaxis)
        points = np.vstack([xgrid.ravel(), ygrid.ravel()]).T

        return self.eval(points).reshape((y_resolution, resolution))
        
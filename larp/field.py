from multiprocessing import Pool
from typing import List, Optional, Tuple, Union

import numpy as np
import larp.fn as lpf
from larp.types import FieldScaleTransform, RGJDict, FieldSize, Point, RepulsionVectorsAndRef

"""
Author: Josue N Rivera

x are assumed to be a list of point coordinates
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

    def __init__(self, coordinates:Union[np.ndarray, List[Point], Point], repulsion:np.ndarray, **kwargs) -> None:
        self.coordinates = np.array(coordinates)
        self.repulsion = np.array(repulsion)
        self.inv_repulsion = np.linalg.inv(self.repulsion)
        self.eye_repulsion = np.eye(len(self.repulsion))
        evals, evects = np.linalg.eig(self.inv_repulsion)
        self.half_inv_repulsion = evects * np.sqrt(evals) @ np.linalg.inv(evects)
        self.inv_repulsion2x = self.inv_repulsion + self.inv_repulsion.T

    def get_dist_matrix(self, scaled=True, inverted=True):

        if inverted and scaled:
            return self.inv_repulsion
        if not scaled:
            return self.eye_repulsion
        
        return self.half_inv_repulsion

    def get_center_point(self) -> np.ndarray:
        return self.coordinates if len(self.coordinates.shape) <= 1 else np.reshape(self.coordinates, (-1, 2)).mean(0)

    def squared_dist(self, x:np.ndarray, scaled=True, inverted=True) -> np.ndarray:
        raise NotImplementedError
    
    def repulsion_vector(self, x:np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def gradient(self, x:np.ndarray, **kwargs):
        repulsion = self.repulsion_vector(x, **kwargs)
        return - self.eval(x=x).reshape(-1, 1) * (repulsion@self.inv_repulsion2x.T)

    def eval(self, x:np.ndarray):
        return np.exp(-self.squared_dist(x))

class PointRGJ(RGJGeometry):
    RGJType = "Point"

    def __init__(self, coordinates: np.ndarray, repulsion: np.ndarray, **kwargs) -> None:
        super().__init__(coordinates, repulsion)

    def repulsion_vector(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return x - self.coordinates
    
    def squared_dist(self, x: np.ndarray, scaled=True, inverted=True) -> np.ndarray:

        x_d_xh = self.repulsion_vector(x=x)
        matrix = self.get_dist_matrix(scaled=scaled, inverted=inverted)
        
        return ((x_d_xh@matrix)*x_d_xh).sum(axis=1)

class LineStringRGJ(RGJGeometry):
    RGJType = "LineString"

    def __init__(self, coordinates: np.ndarray, repulsion: np.ndarray, **kwargs) -> None:
        super().__init__(coordinates, repulsion)
        self.lines_n = len(coordinates)
        self.points_in_line_pair = np.stack([self.coordinates[:-1], self.coordinates[1:]], axis=1)

    def __squared_dist_one_line__(self, x: np.ndarray, line: np.ndarray, scaled=True, inverted=True) -> np.ndarray:

        x_d_g = self.__repulsion_vector_one_line__(x=x, line=line)
        matrix = self.get_dist_matrix(scaled=scaled, inverted=inverted)

        return ((x_d_g@matrix)*x_d_g).sum(1)
    
    def __repulsion_vector_one_line__(self, x: np.ndarray, line: np.ndarray) -> np.ndarray:
        x2_d_x1 = line[1:2] - line[0:1]
        x_d_x1 = x - line[0:1]

        x12dotxx1 = (x2_d_x1*x_d_x1).sum(1, keepdims=True)
        x12dotx12 = (x2_d_x1*x2_d_x1).sum(1, keepdims=True)

        g = line[0] + np.clip(x12dotxx1/x12dotx12, 0.0, 1.0)*(x2_d_x1)
        return x - g
    
    def repulsion_vector(self, x: np.ndarray, min_dist_select:bool = True, **kwargs) -> np.ndarray:
        if self.lines_n > 20:
            p = Pool(5)
            vectors:np.ndarray = p.map(lambda line: self.__repulsion_vector_one_line__(x=x, line=line), self.points_in_line_pair)
        else:
            vectors:np.ndarray = [self.__repulsion_vector_one_line__(x=x, line=line) for line in self.points_in_line_pair]

        vectors = np.stack(vectors, axis=0)
        if min_dist_select:
            vectors = vectors.swapaxes(0, 1)
            dist = (vectors*vectors).sum(-1)
            select = dist.min(1, keepdims=True) == dist
            vectors = vectors[select]
        
        return vectors
    
    def squared_dist(self, x: np.ndarray, scaled=True, inverted=True) -> np.ndarray:
        
        if self.lines_n > 20:
            p = Pool(3)
            dist:np.ndarray = p.map(lambda line: self.__squared_dist_one_line__(x=x, line=line, scaled=scaled, inverted=inverted), self.points_in_line_pair)
        else:
            dist:np.ndarray = [self.__squared_dist_one_line__(x=x, line=line, scaled=scaled, inverted=inverted) for line in self.points_in_line_pair]

        return np.stack(dist, axis=1).min(axis=1)
    
class RectangleRGJ(RGJGeometry):
    RGJType = "Rectangle"

    def __init__(self, coordinates: np.ndarray, repulsion: np.ndarray, **kwargs) -> None:
        super().__init__(coordinates, repulsion)
        self.x1_abs_x2 = np.abs(self.coordinates[0] - self.coordinates[1])

    def repulsion_vector(self, x: np.ndarray, **kwargs) -> np.ndarray:
        
        return 0.5*np.sign(x-self.coordinates[0])*(np.abs(x-self.coordinates[0]) + np.abs(x-self.coordinates[1]) - self.x1_abs_x2)
    
    def squared_dist(self, x: np.ndarray, scaled=True, inverted=True) -> np.ndarray:

        nvector = self.repulsion_vector(x)
        matrix = self.get_dist_matrix(scaled=scaled, inverted=inverted)

        return ((nvector@matrix)*nvector).sum(axis=1)
    
class EllipseRGJ(RGJGeometry):
    RGJType = "Ellipse"
    DEN_ERROR_BUFFER = 1e-6

    def __init__(self, coordinates: np.ndarray, repulsion: np.ndarray, shape: np.ndarray, **kwargs) -> None:
        super().__init__(coordinates, repulsion)
        self.shape = shape
        self.inv_shape = np.linalg.inv(self.shape)

    def repulsion_vector(self, x: np.ndarray, **kwargs) -> np.ndarray:

        x_d_xh = x - self.coordinates
        Binvx = x_d_xh@self.inv_shape.T

        den = np.sqrt((Binvx*Binvx).sum(1, keepdims=True))
        den = np.maximum(den, self.DEN_ERROR_BUFFER)

        return np.maximum(1 - 1/den, 0)*x_d_xh
    
    def squared_dist(self, x: np.ndarray, scaled:bool=True, inverted:bool=True) -> np.ndarray:

        nvector = self.repulsion_vector(x)
        matrix = self.get_dist_matrix(scaled=scaled, inverted=inverted)

        return ((nvector@matrix)*nvector).sum(axis=1) 

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

    def __len__(self)->int:
        return len(self.rgjs)

    def __calculate_center_point__(self):
        return np.array([rgj.get_center_point() for rgj in self.rgjs]).mean(0)
    
    def get_extent(self):
        size2 = self.size[0]/2.0
        return [self.center_point[0] - size2,
                self.center_point[0] + size2,
                self.center_point[1] - size2,
                self.center_point[1] + size2]

    def addRGJ(self, rgj_dict:RGJDict) -> None:
        rgj:RGJGeometry = globals()[rgj_dict["type"]+"RGJ"](**rgj_dict)

        if rgj.RGJType != rgj_dict["type"]:
            raise RuntimeError("RGJ type does not match")
        
        self.rgjs.append(rgj)

        if self.__reload_center:
            self.center_point = self.__calculate_center_point__()

    def delRGJ(self, idx:int) -> None:

        del self.rgjs[idx]
    
    def repulsion_vectors(self, points: Union[np.ndarray, List[Point]], filted_idx:Optional[List[int]] = None, min_dist_select:bool = False, reference_idx = False) -> Union[np.ndarray, RepulsionVectorsAndRef]:
        points = np.array(points)
        filted_idx = filted_idx if not filted_idx is None else list(range(len(self)))

        if reference_idx:
            idxs = []
            repulsion_vectors = []

            for idx in filted_idx:
                vectors = self.rgjs[idx].repulsion_vector(points, min_dist_select=min_dist_select).reshape(-1, 2)

                idxs.extend([idx]*len(vectors))
                repulsion_vectors.append(vectors)

            repulsion_vectors = np.concatenate(repulsion_vectors, axis=0)
            return repulsion_vectors, np.array(idxs, dtype=int)
   
        else:
            rgjs = [self.rgjs[idx] for idx in filted_idx]
            return np.concatenate([rgj.repulsion_vector(points, min_dist_select=min_dist_select).reshape(-1, 2) for rgj in rgjs], axis=0)
        
    def gradient(self, points: Union[np.ndarray, List[Point]]) -> np.ndarray:

        points = np.array(points)
        _, grad_idxs = self.squared_dist(points=points, reference_idx=True)

        grad = np.ones((len(points), 2), dtype=float)
        for idx in set(grad_idxs):
            select = idx == grad_idxs
            grad[select] = self.rgjs[idx].gradient(points[select])

        return grad

    def eval(self, points: Union[np.ndarray, List[Point]], filted_idx:Optional[List[int]] = None) -> np.ndarray:
        points = np.array(points)
        rgjs = [self.rgjs[idx] for idx in filted_idx] if not filted_idx is None else self.rgjs

        return np.max(np.stack([rgj.eval(points) for rgj in rgjs], axis=1), axis=1)
    
    def eval_per(self, points: Union[np.ndarray, List[Point]], idxs:Optional[List[int]] = None) -> np.ndarray:
        if len(points) != len(idxs):
            raise RuntimeError("The number of points doesn't match the number of indexes")
        
        n = len(points)
        idxs = np.array(idxs, dtype=int)
        
        evals = np.ones(n, dtype=float)
        for idx in set(idxs):
            select = idx == idxs
            evals[select] = self.rgjs[idx].eval(points[select])

        return evals
    
    def estimate_route_area(self, route:Union[List[Point], np.ndarray], step=1e-3, n=0, scale_transform:FieldScaleTransform = lambda x: x) -> float:
        route = np.array(route)

        points, step, _ = lpf.interpolate_along_route(route=route, step=step, n=n, return_step_n=True)
        points = points if n <= 0 else points[:-1]

        f_eval = scale_transform(self.eval(points=points))

        return f_eval.sum()*step
    
    def squared_dist(self, points:Union[np.ndarray, List[Point]], filted_idx:Optional[List[int]] = None, scaled=True, inverted=True, reference_idx = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        points = np.array(points)

        dists = self.squared_dist_list(points=points, filted_idx=filted_idx, scaled=scaled, inverted=inverted)
        if reference_idx:
            min_idxs = np.argmin(dists, axis=1)
            return dists[np.arange(len(dists)), min_idxs], min_idxs

        return np.min(dists, axis=1)
    
    def squared_dist_list(self, points:Union[np.ndarray, List[Point]], filted_idx:Optional[List[int]] = None, scaled=True, inverted=True) -> np.ndarray:
        points = np.array(points)
        rgjs = [self.rgjs[idx] for idx in filted_idx] if not filted_idx is None else self.rgjs

        return np.stack([rgj.squared_dist(points, scaled=scaled, inverted=inverted) for rgj in rgjs], axis=1)
    
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
        
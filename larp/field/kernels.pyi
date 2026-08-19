"""Type stubs for larp.field.kernels (compiled from kernels.pyx)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

def segment_repulsion_vectors(
    x: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    v_dot_v: npt.NDArray[np.float64],
    metric: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...

def point_in_rings(
    pts: npt.NDArray[np.float64],
    outer: npt.NDArray[np.float64],
    holes: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.uint8]: ...

def point_obstacles_eval_max(
    x: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    inv_repulsions: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...

def quadtree_find_leaf(
    pts: npt.NDArray[np.float64],
    cx: npt.NDArray[np.float64],
    cy: npt.NDArray[np.float64],
    is_leaf: npt.NDArray[np.uint8],
    child_tl: npt.NDArray[np.int32],
    child_tr: npt.NDArray[np.int32],
    child_bl: npt.NDArray[np.int32],
    child_br: npt.NDArray[np.int32],
    max_depth: int,
) -> npt.NDArray[np.int32]: ...

def quadtree_find_chain(
    pts: npt.NDArray[np.float64],
    cx: npt.NDArray[np.float64],
    cy: npt.NDArray[np.float64],
    is_leaf: npt.NDArray[np.uint8],
    child_tl: npt.NDArray[np.int32],
    child_tr: npt.NDArray[np.int32],
    child_bl: npt.NDArray[np.int32],
    child_br: npt.NDArray[np.int32],
    max_depth: int,
    max_cols: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...

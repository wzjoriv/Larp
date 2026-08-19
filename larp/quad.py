"""
QuadTree/QuadNode now live in larp.field_cy.quadtree (Cython cdef classes).
This module remains as a backward-compatible import path for existing code
that does `import larp.quad` / `from larp.quad import QuadTree, QuadNode`.
"""
from larp.field_cy.quadtree import QuadTree, QuadNode

__all__ = ["QuadTree", "QuadNode"]

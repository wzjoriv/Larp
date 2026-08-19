cdef class RGJGeometry:
    cdef public object coordinates
    cdef public object repulsion
    cdef public object inv_repulsion
    cdef public object eye_repulsion
    cdef public object grad_matrix
    cdef public object properties
    cdef public object bbox


cdef class MultiRGJGeometry(RGJGeometry):
    pass

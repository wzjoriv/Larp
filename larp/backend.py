
import numpy as np
np_o = np
BACKEND = 'NUMPY'

try:
    import cupy as cp
    if cp.cuda.is_available():
        import cupy as np
        cp_o = np
        BACKEND = "CUPY"
    else:
        print("GPU not available, using NumPy")
except ImportError:
    print("CuPy not installed, using NumPy")

def to_numpy(x):
    if hasattr(x, "get"):
        return x.get()
    elif isinstance(x, list):
        return [to_numpy(el) for el in x]
    else:
        return x  # Either NumPy array or scalar
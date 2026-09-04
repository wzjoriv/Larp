import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OSQP_ALGEBRA_BACKEND"] = "builtin"

from larp.field import *       # Cython-backed RiskField (merged w/ quad decomposition) + GeoJSON geometries
from larp.fn import *
from larp.dynamics import Dynamics

import larp.pp as pp, larp.tp as tp
import larp.field as field
import larp.dynamics as dynamics

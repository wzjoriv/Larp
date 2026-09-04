import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OSQP_ALGEBRA_BACKEND"] = "builtin"

from larp.field import *       # Cython-backed RiskField (merged w/ quad decomposition) + GeoJSON geometries
from larp.fn import *
from larp.quad import QRiskField
from larp.dynamics import Dynamics

import larp.quad as quad, larp.pp as pp, larp.tp as tp
import larp.dynamics as dynamics
from typing import Dict, List, Optional, Tuple
import numpy as np
from itertools import chain

"""
Author: Josue N Rivera
"""
        
class Dynamics():

    def __init__(self, 
                 constants:Optional[Dict] = None,
                 state_derivative_orders:List[int] = [1],
                 control_derivative_orders:List[int] = [0]) -> None:

        self.constants = constants

        # Maximum derivitive order for each primitive state needed to represent the system's state vector (same for control)
        
        self.state_derivative_orders = np.array(state_derivative_orders, dtype=int)
        self.control_derivative_orders = np.array(control_derivative_orders, dtype=int)
        
        # Privitive state devitive order for each system state
        self.first_state_orders = np.concatenate([np.arange(i+1, dtype=int) for i in self.state_derivative_orders])
        self.first_control_orders = np.concatenate([np.arange(i+1, dtype=int) for i in self.control_derivative_orders])
        
        # One order higher of the max derivitive order for each primitive state needed to form the ode
        self.highest_state_order = max(self.state_derivative_orders)
        self.highest_control_order = max(self.control_derivative_orders)
        
        self.state_primitive_mask = np.array(list(chain(*[[True]+[False]*i for i in state_derivative_orders])), dtype=bool).reshape(-1)
        self.control_primitive_mask = np.array(list(chain(*[[True]+[False]*i for i in control_derivative_orders])), dtype=bool).reshape(-1)

        self.primitive_state_n, self.primitive_control_n = (len(state_derivative_orders), len(control_derivative_orders))

        self.first_order_state_n, self.first_order_control_n = (sum(state_derivative_orders) + self.primitive_state_n, sum(control_derivative_orders) + self.primitive_control_n)
    
    def split_first(self, first:np.ndarray):

        return tuple([first[:, i:i+1] for i in range(first.size(1))])
    
    def f(self,
          time: np.ndarray, 
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        raise NotImplementedError
    
    def dfdx(self,
          time: np.ndarray, 
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        raise NotImplementedError
    
    def dfdu(self,
          time: np.ndarray, 
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        raise NotImplementedError
        
    def first_state_names(self) -> List[str]:
        "Returns a text label for state names"

        orders = self.state_derivative_orders

        return [f'x_{{{orders[o_idx]}}}^{{[{i}]}}' for o_idx in range(orders) for i in range(orders[o_idx]+1)]
        
    def first_control_names(self) -> List[str]:
    
        orders = self.control_derivative_orders
        return [f'u_{{{orders[o_idx]}}}^{{[{i}]}}' for o_idx in range(orders) for i in range(orders[o_idx]+1)]
        
    def first_names(self) -> Tuple[List[str], List[str]]:
                   
        return self.first_state_names(), self.first_control_names()
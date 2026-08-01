from dataclasses import dataclass
import numpy as np
import itertools
from functools import lru_cache, reduce
from numpy.typing import NDArray

@dataclass # NodeParams
class Node:
    '''Node parameters'''
    mean_r: float
    var_r: float
    var_x: float
    def post_var(self,t):
        return (self.var_r*self.var_x)/(self.var_x+t*self.var_r)
    
def var_trans(p: Node, t: float, dt: float):
    pv = p.post_var(t)
    fac = (p.var_r / (p.var_r*(t+dt) + p.var_x))**2
    return fac * (p.var_x + pv*dt) * dt

def gauss1d(scale: NDArray, var: float):
    w=np.exp(-0.5*(scale/np.sqrt(var))**2)
    return w/w.sum()

@lru_cache(maxsize=64) #
def sep_kernel(scale_min: float, scale_max: float, res: int,
    var_tuple: tuple[float,...]):
    scale = np.linspace(scale_min, scale_max, res)
    k = reduce(lambda a,b: np.multiply.outer(a,b),
    [gauss1d(scale,v) for v in var_tuple])
    return k



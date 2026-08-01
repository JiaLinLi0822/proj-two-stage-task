import numpy as np

from util import Node

def get_params(which_tree: int = None, param_input: dict = None) -> dict:
    """
    Parameters for value-based decision making
    
    Args:
        which_tree (int, optional): Tree type (1 or 2). Defaults to None.
        param_input (dict, optional): Input parameter values. Defaults to None.
    
    Returns:
        dict: Dictionary containing all parameters
    """

    zs = 100 # belief grid resolution
    z_max = 2.0
    dt = 0.05 # time step
    T = 5.0 # Total time horizon
    ts = int(T/dt) + 1 # time grid
    rho = 0.0 # reward rate
    cost = 0.5 # cost of waiting
    t_null = 0.25 # time to nullify the effect of the first stage
    # t_null2 = 0.15 # time to nullify the effect of the second stage

    base_node = Node(mean_r=0.0, var_r=1.0, var_x=1.0) # base node parameters

    utility = lambda x: x
    
    Params = {
        "T": T,
        "dt": dt,
        "base_node": base_node,
        "zs": zs,
        "z_max": z_max,
        "ts": ts,
        "which_tree": which_tree,
        "utility": utility,
        "rho": rho,
        "cost": cost,
        "t_null": t_null,
        "rho": rho,
    }

    if which_tree == 1:
        Params["which_tree"] = "1"
    elif which_tree == 2:
        Params["which_tree"] = "2"
    else:
        raise ValueError("Invalid tree type")

    return Params
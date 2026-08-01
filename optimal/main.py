import numpy as np
from tqdm import tqdm
from getParams import get_params
from util import var_trans, sep_kernel
from scipy import ndimage, optimize

def solve_leaf(params):

    zs, ts, z_max, dt, rho, base_node, utility, cost, t_null2 = (
        params["zs"], params["ts"], params["z_max"], params["dt"], 
        params["rho"], params["base_node"], params["utility"], params["cost"], params["t_null2"]
    )

    Z1 = np.linspace(-z_max, z_max, zs)
    Z2 = np.linspace(-z_max, z_max, zs)
    Z1_grid, Z2_grid = np.meshgrid(Z1, Z2)

    V = np.empty((zs, zs, ts))
    D = np.empty((zs, zs, ts))

    # terminal: must choose 
    V[:, :, ts - 1] = np.maximum(utility(Z1_grid), utility(Z2_grid)) - rho * t_null2
    D[:, :, ts - 1] = (utility(Z1_grid) >= utility(Z2_grid)).astype(int) + 2 # 2 for A, 3 for B

    for t in tqdm(range(ts - 2, -1, -1)):

        time = t * dt
        var_A = var_trans(base_node, time, dt)
        var_B = var_trans(base_node, time, dt)

        kernel = sep_kernel(-z_max, z_max, zs, (var_A, var_B))
        
        future = ndimage.convolve(V[:, :, t+1], kernel, mode='nearest')
        wait = future - (rho + cost) * dt
        chooseA = utility(Z1_grid) - rho * t_null2
        chooseB = utility(Z2_grid) - rho * t_null2
        Q = np.stack([wait, chooseA, chooseB], axis=0)  

        V[:, :, t] = np.max(Q, axis=0)
        D[:, :, t] = np.argmax(Q, axis=0) + 1 # 1 for wait, 2 for A, 3 for B

    return V, D

def solve_root(params):
    
    zs, ts, z_max, dt, rho, base_node, utility, cost, t_null1 = (
        params["zs"], params["ts"], params["z_max"], params["dt"], 
        params["rho"], params["base_node"], params["utility"], params["cost"], params["t_null1"]
    )

    r_coords = np.linspace(-z_max, z_max, zs)
    rL, rR, rLL, rLR, rRL, rRR = np.meshgrid(r_coords, r_coords, r_coords, r_coords, r_coords, r_coords)


    # V1, V2 stores the value function for two stages
    V1 = np.empty((zs, zs, zs, zs, ts))
    V2 = np.empty((zs, zs, zs, zs, ts))
    D1 = np.empty((zs, zs, zs, zs, ts))
    D2 = np.empty((zs, zs, zs, zs, ts))

    V1[:, :, :, :, ts - 1] = np.maximum(utility(Z1_grid), utility(Z2_grid)) - rho * t_null1
    V2[:, :, :, :, ts - 1] = np.maximum(utility(Z1_grid), utility(Z2_grid)) - rho * t_null1

    D1[:, :, :, :, ts - 1] = np.argmax(np.array([utility(Z1_grid), utility(Z2_grid)]), axis=0).astype(np.uint8) + 1
    D2[:, :, :, :, ts - 1] = np.argmax(np.array([utility(Z1_grid), utility(Z2_grid)]), axis=0).astype(np.uint8) + 1

    for t in tqdm(range(ts - 2, -1, -1)):

        time = t * dt
        var_A = var_trans(base_node, time, dt)
        var_B = var_trans(base_node, time, dt)

        kernel = sep_kernel(-z_max, z_max, zs, (var_A, var_B))

        # Stage 1
        future = ndimage.convolve(V1[:, :, t+1], kernel, mode='nearest')
        wait = future - (rho + cost) * dt
        #
        chooseA = V2[:, :, t+1, 0, 0] - rho * t_null2
        chooseB = V2[:, :, t+1, 1, 1] - rho * t_null2
        Q = np.stack([wait, chooseA, chooseB], axis=0)  

        V1[:, :, t] = np.max(Q, axis=0)
        D1[:, :, t] = np.argmax(Q, axis=0) + 1 # 1 for wait, 2 for A, 3 for B
    
    












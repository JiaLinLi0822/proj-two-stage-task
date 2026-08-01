import numpy as np
from scipy import ndimage
from tqdm import tqdm
from util import var_trans, sep_kernel
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from getParams import get_params
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import norm


def gaussian_kernel_2d(z1_max, z2_max, zs, var1, var2, rho):
    """
    构造一个归一化的二维高斯平移核，协方差矩阵为
        [[var1,      rho*sqrt(var1*var2)],
         [rho*sqrt(var1*var2),    var2 ]]
    用于模拟在 dt 时间内，后验 mean (z1,z2) 的高斯预测分布。
    """
    x = np.linspace(-z1_max, z1_max, zs)
    y = np.linspace(-z2_max, z2_max, zs)
    X, Y = np.meshgrid(x, y, indexing='xy')

    cov12 = rho * np.sqrt(var1 * var2)
    cov = np.array([[var1, cov12],
                    [cov12, var2]])
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    # 2D Gaussian PDF
    expo = (
        inv_cov[0, 0] * X**2
      + 2 * inv_cov[0, 1] * X * Y
      + inv_cov[1, 1] * Y**2
    )
    kernel = np.exp(-0.5 * expo) / (2 * np.pi * np.sqrt(det_cov))

    # 归一化
    return kernel / kernel.sum()

def clark(mu1, mu2, sigma1, sigma2, rho):
    """
    mu1, mu2: 两条路径的后验均值
    sigma1, sigma2: 两条路径的后验标准差
    rho: 两条路径的相关系数
    返回 E[ max(Z1, Z2) ]
    """
    θ = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
    δ = (mu1 - mu2) / θ
    return (
        mu1 * norm.cdf(δ)
      + mu2 * norm.cdf(-δ)
      + θ   * norm.pdf(δ)
    )

def solve_stage2(params):

    zs, ts, z_max, dt, base_node, utility, cost, t_null = (
        params["zs"], params["ts"], params["z_max"], params["dt"], 
        params["base_node"], params["utility"], 
        params["cost"], params["t_null"]
    )

    Z1 = np.linspace(-z_max, z_max, zs)
    Z2 = np.linspace(-z_max, z_max, zs)
    Z1_grid, Z2_grid = np.meshgrid(Z1, Z2)

    V = np.empty((zs, zs, ts))
    D = np.empty((zs, zs, ts))

    # terminal: must choose 
    V[:, :, ts - 1] = np.maximum(utility(Z1_grid), utility(Z2_grid)) - t_null
    D[:, :, ts - 1] = (utility(Z1_grid) >= utility(Z2_grid)).astype(int) + 2 # 2 for A, 3 for B

    for t in tqdm(range(ts - 2, -1, -1)):

        time = t * dt
        var_A = var_trans(base_node, time, dt)
        var_B = var_trans(base_node, time, dt)

        kernel = sep_kernel(-z_max, z_max, zs, (var_A, var_B))
        
        future = ndimage.convolve(V[:, :, t+1], kernel, mode='nearest')
        wait = future - cost * dt
        chooseA = utility(Z1_grid) - t_null
        chooseB = utility(Z2_grid) - t_null
        Q = np.stack([wait, chooseA, chooseB], axis=0)  

        V[:, :, t] = np.max(Q, axis=0)
        D[:, :, t] = np.argmax(Q, axis=0) + 1 # 1 for wait, 2 for A, 3 for B

    return V, D

# def solve_stage1(params):
#     """
#     动态规划求解：在拥有 (z1,z2) 后验 mean
#     随时间演化下，何时停止观测 vs. 立刻选择 max 的最优策略。

#     params 中应包含：
#       zs        : 网格维度（z1,z2 各方向上格点数）
#       ts        : 时间步数
#       z_max     : z1,z2 的范围 [-z_max, +z_max]
#       dt        : 时间步长
#       rho       : z1,z2 的相关系数（后验 covariance 归一化后）
#       base_node : 用于计算 var_trans 的基准参数
#       utility  : 函数 u(z)，给出即时选择 A/B 的效用
#       cost     : 单位时间观测成本
#       t_null    : 立即选择时的固定延迟成本
#     """
#     zs, ts, z_max, dt = (
#         params["zs"], params["ts"],
#         params["z_max"], params["dt"]
#     )
#     rho        = params["rho"]
#     base_node  = params["base_node"]
#     utility    = params["utility"]
#     cost       = params["cost"]
#     t_null     = params["t_null"]

#     # 构造 z1, z2 网格
#     grid = np.linspace(-z_max, z_max, zs)
#     Z_LL, Z_LR, Z_RL, Z_RR = np.meshgrid(grid, grid, grid, grid, indexing='xy')

#     # 初始化 value & decision arrays
#     V = np.empty((zs, zs, zs, zs, ts))
#     D = np.empty((zs, zs, zs, zs, ts), dtype=np.int8)

#     # 终态：必须选择 A 或 B
#     V[:, :, :, :, ts - 1] = np.maximum(utility(Z_LL), utility(Z_LR), utility(Z_RL), utility(Z_RR)) - t_null
#     # action: 2 表示选 A, 3 表示选 B
#     D[:, :, :, :, ts - 1] = 

#     # 反向迭代
#     for t in tqdm(range(ts - 1, -1, -1)):

#         vL = V_sub_L[:, :, t]
#         vR = V_sub_R[:, :, t]

#         if t == ts - 1:
#             V[:, :, t] = np.maximum(vL, vR)
#             D[:, :, t] = (vL >= vR).astype(int) + 2
#             continue

#         # 计算在 dt 时间内预测后验 mean 演化所带来的增量方差
#         time = t * dt

#         # 左子树
#         v_L0 = var_trans(base_node, time, dt)
#         v_LL = var_trans(base_node, time, dt)
#         v_LR = var_trans(base_node, time, dt)

#         # 右子树
#         v_R0 = var_trans(base_node, time, dt)
#         v_RL = var_trans(base_node, time, dt)
#         v_RR = var_trans(base_node, time, dt)

#         var_LL = v_L0 + v_LL
#         var_LR = v_L0 + v_LR
#         rho_L = v_L0 / np.sqrt((v_L0+v_LL)*(v_L0+v_LR))

#         var_RL = v_R0 + v_RL
#         var_RR = v_R0 + v_RR
#         rho_R = v_R0 / np.sqrt((v_R0+v_RL)*(v_R0+v_RR))

#          # 1) 左子树卷积核
#         K_L = gaussian_kernel_2d(z_max, z_max, zs,
#                                 var1=v_L0+v_LL,
#                                 var2=v_L0+v_LR,
#                                 rho=rho_L)

#         # 2) 右子树卷积核
#         K_R = gaussian_kernel_2d(z_max, z_max, zs,
#                                 var1=v_R0+v_RL,
#                                 var2=v_R0+v_RR,
#                                 rho=rho_R)
        
#         V1 = ndimage.convolve(V[:, :, t + 1], K_L[:, :, None, None], mode='nearest')
#         V2 = ndimage.convolve(V1, K_R[None, None, :, :], mode='nearest')

#         future = V2

#         # 选项一：继续观测（等待）
#         Q_wait = future - cost * dt

#         # (a) 立即Enter L：Clark((z_LL,z_LR); var, rho) – t_null
#         #    z_LL, z_LR 就用 vL 网格
#         mu_LL, mu_LR = vL, vL  # shape (zs,zs)
#         sigma_LL, sigma_LR = np.sqrt(var_LL), np.sqrt(var_LR)
#         Q_L = clark(mu_LL, mu_LR, sigma_LL, sigma_LR, rho_L) - t_null

#         # (b) 立即Enter R：同理
#         mu_RL, mu_RR = vR, vR
#         sigma_RL, sigma_RR = np.sqrt(var_RL), np.sqrt(var_RR)
#         Q_R = clark(mu_RL, mu_RR, sigma_RL, sigma_RR, rho_R) - t_null

#         # 合并三个动作的 Q 值，并取最优
#         Q = np.stack([Q_wait, Q_L, Q_R], axis=0)  # (3,zs,zs)
#         V[:,:,t] = Q.max(axis=0)
#         D[:,:,t] = Q.argmax(axis=0) + 1  # 1=Wait,2=Left,3=Right
#         # 1 → wait, 2 → choose A, 3 → choose B

#     return V, D

def solve_stage1(params):
    """
    对于四条路径的两层树结构，动态规划求解 (z_LL, z_LR, z_RL, z_RR) 的最优策略。

    params 中应包含：
      zs        : 网格维度（每维的格点数）
      ts        : 时间步数
      z_max     : z 的范围 [-z_max, +z_max]
      dt        : 时间步长
      rho_L     : 左子树内部路径的相关系数
      rho_R     : 右子树内部路径的相关系数
      base_node : 用于 var_trans 的基准节点
      utility   : 路径效用函数（暂未使用）
      cost      : 单位时间观测成本
      t_null    : 立即选择时的延迟成本
    """
    import numpy as np
    from scipy.stats import norm, multivariate_normal
    from scipy import ndimage
    from tqdm import tqdm

    zs = params["zs"]
    ts = params["ts"]
    z_max = params["z_max"]
    dt = params["dt"]
    rho_L = params.get("rho_L", params.get("rho", 0.0))
    rho_R = params.get("rho_R", params.get("rho", 0.0))
    base_node = params["base_node"]
    utility = params.get("utility")
    cost = params["cost"]
    t_null = params["t_null"]

    # 构建网格
    grid = np.linspace(-z_max, z_max, zs)
    Z_LL, Z_LR, Z_RL, Z_RR = np.meshgrid(grid, grid, grid, grid, indexing='ij')

    # 预构 pos2 用于 2D 核计算
    X, Y = np.meshgrid(grid, grid, indexing='ij')
    pos2 = np.stack([X, Y], axis=-1)  # shape (zs, zs, 2)

    V = np.empty((zs, zs, zs, zs, ts))
    D = np.empty((zs, zs, zs, zs, ts), dtype=np.int8)

    def clark_np(mu1, mu2, sigma1, sigma2, rho):
        theta = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
        delta = (mu1 - mu2) / theta
        return (
            mu1 * norm.cdf(delta)
            + mu2 * norm.cdf(-delta)
            + theta * norm.pdf(delta)
        )

    # 终止时刻：只可选择左/右子树
    left_term = clark_np(Z_LL, Z_LR, 1, 1, rho_L)
    right_term = clark_np(Z_RL, Z_RR, 1, 1, rho_R)
    V[..., ts-1] = np.maximum(left_term, right_term) - t_null
    D[..., ts-1] = np.where(left_term >= right_term, 2, 3)  # 2=left, 3=right

    # 反向动态规划
    for t in tqdm(range(ts-2, -1, -1)):
        time = t * dt
        # 各路径的后验方差演化
        s_LL = var_trans(base_node, time, dt)
        s_LR = var_trans(base_node, time, dt)
        s_RL = var_trans(base_node, time, dt)
        s_RR = var_trans(base_node, time, dt)

        # 构建左右子树的协方差子块
        cov_L = np.array([[s_LL, rho_L * np.sqrt(s_LL * s_LR)],
                          [rho_L * np.sqrt(s_LL * s_LR), s_LR]])
        cov_R = np.array([[s_RL, rho_R * np.sqrt(s_RL * s_RR)],
                          [rho_R * np.sqrt(s_RL * s_RR), s_RR]])

        # 计算 2D 卷积核
        rv_L = multivariate_normal(mean=[0, 0], cov=cov_L)
        K_L = rv_L.pdf(pos2)
        K_L /= K_L.sum()
        rv_R = multivariate_normal(mean=[0, 0], cov=cov_R)
        K_R = rv_R.pdf(pos2)
        K_R /= K_R.sum()

        # 第 1 步：沿 (LL, LR) 维度做 2D 卷积
        tmp = np.empty((zs, zs, zs, zs))
        for i_rl in range(zs):
            for j_rr in range(zs):
                tmp[:, :, i_rl, j_rr] = ndimage.convolve(
                    V[:, :, i_rl, j_rr, t+1], K_L, mode='nearest'
                )

        # 第 2 步：沿 (RL, RR) 维度做 2D 卷积
        future = np.empty_like(tmp)
        for i_ll in range(zs):
            for j_lr in range(zs):
                future[i_ll, j_lr, :, :] = ndimage.convolve(
                    tmp[i_ll, j_lr, :, :], K_R, mode='nearest'
                )

        # 更新 Q 值
        Q_wait = future - cost * dt
        Q_left = clark_np(Z_LL, Z_LR, np.sqrt(s_LL), np.sqrt(s_LR), rho_L) - t_null
        Q_right = clark_np(Z_RL, Z_RR, np.sqrt(s_RL), np.sqrt(s_RR), rho_R) - t_null

        Q_all = np.stack([Q_wait, Q_left, Q_right], axis=0)
        V[..., t] = np.max(Q_all, axis=0)
        D[..., t] = np.argmax(Q_all, axis=0) + 1  # 1=wait, 2=left, 3=right

    return V, D

def solve_tree(params):
    V, D = solve_stage2(params)
    V_sub_L, V_sub_R = V, V
    V, D = solve_stage1(params, V_sub_L, V_sub_R)
    return V, D


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def plot_value_slice(V, grid, t=0, slice_dims=('RL','RR'),
                     slice_idx=None, surface3d=False, cmap='viridis', n_levels=50):
    """
    在 V[LL, LR, RL, RR, t] 上固定 slice_dims（'RL','RR' 或 'LL','LR'），
    在剩余两个维度上画热图或 3D 曲面。

    Parameters
    ----------
    V : ndarray, shape (zs,zs,zs,zs,ts)
    grid : 1d array of length zs
    t : int, 时间步
    slice_dims : tuple, 待固定的维度名 ('RL','RR') 或 ('LL','LR')
    slice_idx : tuple of ints, 固定维度的索引；若 None，则取中点
    surface3d : bool, 是否画 3D 曲面，否则画等高线热图
    cmap : str or Colormap
    n_levels : int, 等高线层数
    """
    zs = grid.size
    # 维度顺序：0=LL,1=LR,2=RL,3=RR
    dim_map = {'LL':0, 'LR':1, 'RL':2, 'RR':3}
    # 剩余两个维度
    all_dims = ['LL','LR','RL','RR']
    other_dims = [d for d in all_dims if d not in slice_dims]
    # 切片索引
    if slice_idx is None:
        slice_idx = (zs//2, zs//2)
    idx = { dim_map[slice_dims[0]]: slice_idx[0],
            dim_map[slice_dims[1]]: slice_idx[1],
            4: t }  # time 轴

    # 构造切片视图
    slicer = [slice(None)]*5
    slicer[4] = t
    slicer[dim_map[slice_dims[0]]] = slice_idx[0]
    slicer[dim_map[slice_dims[1]]] = slice_idx[1]
    data2d = V[tuple(slicer)]  # shape (zs, zs)

    # 构造两个可视化轴的网格
    X, Y = np.meshgrid(grid, grid, indexing='ij')

    if surface3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, data2d, cmap=cmap)
        ax.set_xlabel(other_dims[0])
        ax.set_ylabel(other_dims[1])
        ax.set_zlabel('Value')
        ax.set_title(f"V at t={t}, {slice_dims}={slice_idx}")
    else:
        plt.figure()
        cf = plt.contourf(X, Y, data2d, levels=n_levels, cmap=cmap)
        plt.colorbar(cf, label='Value')
        plt.xlabel(other_dims[0])
        plt.ylabel(other_dims[1])
        plt.title(f"V heatmap at t={t}, {slice_dims}={slice_idx}")
    plt.tight_layout()
    plt.show()


def plot_policy_slice(D, grid, t=0, slice_dims=('RL','RR'),
                      slice_idx=None, cmap=None):
    """
    类似 plot_value_slice，但用于策略 D[LL,LR,RL,RR,t] 的离散决策等高图。
    """
    zs = grid.size
    dim_map = {'LL':0, 'LR':1, 'RL':2, 'RR':3}
    all_dims = ['LL','LR','RL','RR']
    other_dims = [d for d in all_dims if d not in slice_dims]
    if slice_idx is None:
        slice_idx = (zs//2, zs//2)

    slicer = [slice(None)]*5
    slicer[4] = t
    slicer[dim_map[slice_dims[0]]] = slice_idx[0]
    slicer[dim_map[slice_dims[1]]] = slice_idx[1]
    data2d = D[tuple(slicer)]

    X, Y = np.meshgrid(grid, grid, indexing='ij')

    # 默认 colormap： wait(1), left(2), right(3)
    if cmap is None:
        cmap = get_cmap('Set3', 3)
    norm = BoundaryNorm([0.5,1.5,2.5,3.5], ncolors=3)

    plt.figure()
    cf = plt.contourf(X, Y, data2d, levels=[0.5,1.5,2.5,3.5],
                      cmap=cmap, norm=norm)
    cbar = plt.colorbar(cf, ticks=[1,2,3])
    cbar.set_ticklabels(['Wait','Left','Right'])
    plt.xlabel(other_dims[0])
    plt.ylabel(other_dims[1])
    plt.title(f"Policy at t={t}, {slice_dims}={slice_idx}")
    plt.tight_layout()
    plt.show()


def animate_value_slice(V, grid, slice_dims=('RL','RR'),
                        slice_idx=None, interval=200, surface3d=False):
    """
    动画版 value 函数切片随时间的变化。
    返回 FuncAnimation 对象。
    """
    zs, _, _, _, ts = V.shape
    # 复用上面 plot_value_slice 的逻辑，每帧调用
    fig = plt.figure()
    if surface3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    def update(frame):
        ax.clear()
        plot_value_slice(V, grid, t=frame,
                         slice_dims=slice_dims,
                         slice_idx=slice_idx,
                         surface3d=surface3d)
        return ax.collections if not surface3d else ax

    ani = FuncAnimation(fig, update, frames=ts, interval=interval, blit=False)
    return ani


def animate_policy_slice(D, grid, slice_dims=('RL','RR'),
                         slice_idx=None, interval=200):
    """
    动画版策略切片随时间的变化。
    """
    zs, _, _, _, ts = D.shape
    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()
        plot_policy_slice(D, grid, t=frame,
                          slice_dims=slice_dims,
                          slice_idx=slice_idx)
        return ax.collections

    ani = FuncAnimation(fig, update, frames=ts, interval=interval, blit=False)
    return ani

def plot_policy_avg(D, grid, t, dt=None, ax=None):
    """
    Plot the policy at time t on the 2D plane:
      x = (LL + LR) / 2
      y = (RL + RR) / 2
    D has shape (zs, zs, zs, zs, ts), grid is 1D array of length zs.
    """
    zs, _, _, _, ts = D.shape
    if not (0 <= t < ts):
        raise ValueError(f"t must be in [0, {ts-1}]")

    # build 4D mesh of posterior means
    Z = grid
    Z_LL, Z_LR, Z_RL, Z_RR = np.meshgrid(Z, Z, Z, Z, indexing='ij')
    X = ((Z_LL + Z_LR) / 2.0).ravel()
    Y = ((Z_RL + Z_RR) / 2.0).ravel()
    P = D[..., t].ravel()  # policy codes 1=wait,2=left,3=right

    # scatter
    cmap = ListedColormap(['#66c2a5','#fc8d62','#8da0cb'])  # wait, left, right
    norm = BoundaryNorm([0.5,1.5,2.5,3.5], ncolors=3)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(X, Y, c=P, cmap=cmap, norm=norm, s=5, linewidth=0)
    ax.set_xlabel('(LL + LR) / 2')
    ax.set_ylabel('(RL + RR) / 2')
    ax.set_title(f'Policy at t = {t}' + (f' ({t*dt:.2f}s)' if dt else ''))
    cbar = plt.colorbar(sc, ax=ax, ticks=[1,2,3])
    cbar.ax.set_yticklabels(['Wait','Left','Right'])
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    if ax is None:
        plt.show()
    return ax

def animate_policy_avg(D, grid, dt, interval=200):
    """
    Animate the policy over time on the averaged 2D plane.
      D: (zs, zs, zs, zs, ts)
      grid: array of length zs
      dt: time step size (for labeling)
      interval: ms between frames
    Returns FuncAnimation object.
    """
    zs, _, _, _, ts = D.shape
    # precompute X,Y flattened
    Z = grid
    Z_LL, Z_LR, Z_RL, Z_RR = np.meshgrid(Z, Z, Z, Z, indexing='ij')
    X = ((Z_LL + Z_LR) / 2.0).ravel()
    Y = ((Z_RL + Z_RR) / 2.0).ravel()

    fig, ax = plt.subplots(figsize=(6,6))
    cmap = ListedColormap(['#66c2a5','#fc8d62','#8da0cb'])
    norm = BoundaryNorm([0.5,1.5,2.5,3.5], ncolors=3)
    sc = ax.scatter([], [], s=5, cmap=cmap, norm=norm)
    ax.set_xlabel('(LL + LR) / 2')
    ax.set_ylabel('(RL + RR) / 2')
    ax.set_aspect('equal', 'box')
    cbar = plt.colorbar(sc, ax=ax, ticks=[1,2,3])
    cbar.ax.set_yticklabels(['Wait','Left','Right'])

    def init():
        sc.set_offsets(np.empty((0,2)))
        sc.set_array(np.empty((0,)))
        ax.set_title('')
        return sc,

    def update(frame):
        P = D[..., frame].ravel()
        pts = np.column_stack((X, Y))
        sc.set_offsets(pts)
        sc.set_array(P)
        ax.set_title(f'Policy t = {frame} ({frame*dt:.2f}s)')
        return sc,

    ani = FuncAnimation(fig, update, frames=ts, init_func=init,
                        interval=interval, blit=True)
    plt.tight_layout()
    return ani

if __name__ == "__main__":
    params = get_params(which_tree=1)

    V, D = solve_stage1(params)

    z_max = params["z_max"] 
    zs = params["zs"]
    ts = params["ts"]

    grid = np.linspace(-z_max, z_max, zs)

    # 假设 V.shape == (zs,zs,zs,zs,ts)，grid = np.linspace(...)
    # 1) 看在 RL=RR=中心 的 LL vs LR 面上 t=5 时的曲面：
    # plot_value_slice(V, grid, t=50,
    #                 slice_dims=('RL','RR'),
    #                 slice_idx=(zs//2, zs//2),
    #                 surface3d=True)

    # # 2) 看在 RL=0, RR=末端 的 LL vs LR 面上 t=0 的热图：
    # plot_value_slice(V, grid, t=50,
    #                 slice_dims=('RL','RR'),
    #                 slice_idx=(0, zs-1),
    #                 surface3d=False)

    # 3) 同样地，画策略：
    plot_policy_slice(D, grid, t=90,
                    slice_dims=('LL','LR'),
                    slice_idx=(0, 0))
    
    # print(V[0,0,:,:,:])
    
    ani = animate_policy_slice(D, grid,
                    slice_dims=('LL','LR'),
                    slice_idx=(0, 10))
    ani.save('policy.gif', writer='imagemagick', fps=10)

    # grid = np.linspace(-z_max, z_max, zs)
    # plot_policy_avg(D, grid, t=10, dt=0.05)
    # ani = animate_policy_avg(D, grid, dt=0.05, interval=200)
    # ani.save('policy_evolution.gif', writer='imagemagick')
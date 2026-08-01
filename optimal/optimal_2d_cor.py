import numpy as np
from scipy import ndimage
from tqdm import tqdm
from util import var_trans, sep_kernel
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from getParams import get_params
from matplotlib.colors import ListedColormap, BoundaryNorm
from copy import deepcopy
from collections import Counter


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


def solve_gaussian_dp(params):
    """
    动态规划求解：在拥有 (z1,z2) 后验 mean
    随时间演化下，何时停止观测 vs. 立刻选择 max 的最优策略。

    params 中应包含：
      zs        : 网格维度（z1,z2 各方向上格点数）
      ts        : 时间步数
      z_max     : z1,z2 的范围 [-z_max, +z_max]
      dt        : 时间步长
      rho       : z1,z2 的相关系数（后验 covariance 归一化后）
      base_node : 用于计算 var_trans 的基准参数
      utility  : 函数 u(z)，给出即时选择 A/B 的效用
      cost     : 单位时间观测成本
      t_null    : 立即选择时的固定延迟成本
    """
    zs, ts, z_max, dt = (
        params["zs"], params["ts"],
        params["z_max"], params["dt"]
    )
    rho        = params["rho"]
    base_node  = params["base_node"]
    utility    = params["utility"]
    cost       = params["cost"]
    t_null     = params["t_null"]

    # 构造 z1, z2 网格
    Z1 = np.linspace(-z_max, z_max, zs)
    Z2 = np.linspace(-z_max, z_max, zs)
    Z1_grid, Z2_grid = np.meshgrid(Z1, Z2, indexing='xy')

    # 初始化 value & decision arrays
    V = np.empty((zs, zs, ts))
    D = np.empty((zs, zs, ts), dtype=np.int8)

    # 终态：必须选择 A 或 B
    V[:, :, ts - 1] = np.maximum(utility(Z1_grid),
                                 utility(Z2_grid)) - t_null
    # action: 2 表示选 A, 3 表示选 B
    D[:, :, ts - 1] = (utility(Z1_grid) >= utility(Z2_grid)).astype(int) + 2

    # 反向迭代
    for t in tqdm(range(ts - 2, -1, -1)):
        # 计算在 dt 时间内预测后验 mean 演化所带来的增量方差
        time = t * dt
        var1 = var_trans(base_node, time, dt)
        var2 = var_trans(base_node, time, dt)

        # 构造相关高斯核
        kernel = gaussian_kernel_2d(z_max, z_max, zs, var1, var2, rho)

        # 卷积得到下一步 value 的预测期望
        future = ndimage.convolve(V[:, :, t + 1], kernel, mode='nearest')

        # 选项一：继续观测（等待）
        Q_wait = future - cost * dt

        # 选项二：立即选 A/B
        Q_A = utility(Z1_grid) - t_null
        Q_B = utility(Z2_grid) - t_null

        # 合并三个动作的 Q 值，并取最优
        Q_stack = np.stack([Q_wait, Q_A, Q_B], axis=0)
        V[:, :, t] = Q_stack.max(axis=0)
        D[:, :, t] = Q_stack.argmax(axis=0) + 1
        # 1 → wait, 2 → choose A, 3 → choose B

    return V, D

# --- Plotting ---
def plot_value_function(V, R1_grid, R2_grid, t=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(R1_grid, R2_grid, V[:, :, t])
    plt.show()

# plot in 2d space
def plot_decision_function(D, R1_grid, R2_grid, t=0):
    
    # use different colors for different decisions
    plt.figure()
    plt.contourf(R1_grid, R2_grid, D[:, :, t], levels=3, cmap='viridis')
    plt.colorbar()
    plt.show()


def animate_value_function(V, R1_grid, R2_grid):
    """
    Create an animation of the value function V over time.
    
    Parameters:
    - V: 3D array of shape (zs, zs, ts)
    - R1_grid, R2_grid: 2D meshgrid arrays of shape (zs, zs)
    - interval: time between frames in ms
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('R1')
    ax.set_ylabel('R2')
    ax.set_zlabel('Value')
    
    # fix z-limits for consistency across frames
    zmin, zmax = np.min(V), np.max(V)
    ax.set_zlim(zmin, zmax)
    
    def update(frame):
        ax.clear()
        ax.set_xlabel('R1')
        ax.set_ylabel('R2')
        ax.set_zlabel('Value')
        ax.set_zlim(zmin, zmax)
        ax.plot_surface(R1_grid, R2_grid, V[:, :, frame], cmap='viridis')
        ax.set_title(f"Value Function at t = {frame}")
        return fig,
    
    ani = animation.FuncAnimation(fig, update, frames=V.shape[2], interval=np.shape(V)[2])
    plt.show()
    return ani

def animate_policy(D, R1_grid, R2_grid):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('R1')
    ax.set_ylabel('R2')
    
    cmap = ListedColormap(['#f2e5f7', '#b3cde3', '#fbb4ae'])  # wait, item1, item2
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    decision_labels = ['Wait', 'Choose item1', 'Choose item2']

    # Create initial contour plot
    cf = ax.contourf(
        R1_grid, R2_grid, D[:, :, 0],
        levels=[0.5, 1.5, 2.5, 3.5],
        cmap=cmap, norm=norm
    )
    cbar = plt.colorbar(cf, ax=ax, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels(decision_labels)
    cbar.set_label('Decision')
    ax.set_title('Policy at t = 0')

    def update(frame):
        # Clear previous contour
        for coll in ax.collections:
            coll.remove()
        
        # Create new contour
        cf = ax.contourf(
            R1_grid, R2_grid, D[:, :, frame],
            levels=[0.5, 1.5, 2.5, 3.5],
            cmap=cmap, norm=norm
        )
        ax.set_title(f'Policy at t = {frame}')
        return cf

    ani = animation.FuncAnimation(
        fig, update, frames=D.shape[2],
        interval=np.shape(D)[2], blit=False
    )
    plt.show()
    return ani

# ---------- collapse policy onto d ----------
def policy_along_d(D_t, d_vals, tol=1e-8):
    """
    For a single t:
      D_t : (zs, zs) policy matrix (1=wait,2=choose1,3=choose2)
      d_vals : (zs, zs) array of d values
    Returns:
      d_sorted : 1D sorted unique d grid
      pol_d    : policy per d (same length), chosen by majority vote over all (z1,z2) with that d (within tol)
    """
    flat_d = d_vals.ravel()
    flat_p = D_t.ravel()

    # Make discrete d-bins by rounding (safer than exact equality)
    d_round = np.round(flat_d / tol)  # integer bins
    uniq_bins = np.unique(d_round)

    d_sorted = []
    pol_d = []

    for b in np.sort(uniq_bins):
        mask = (d_round == b)
        votes = Counter(flat_p[mask])
        # break ties by preferring wait < choose2 < choose1 (or whatever you want)
        # Here: take the most common
        pol = votes.most_common(1)[0][0]
        pol_d.append(pol)
        d_sorted.append(flat_d[mask].mean())  # representative d

    d_sorted = np.array(d_sorted)
    pol_d    = np.array(pol_d)
    order = np.argsort(d_sorted)
    return d_sorted[order], pol_d[order]


def find_boundary(d_sorted, pol_sorted, from_lab, to_lab):
    """
    First transition from from_lab -> to_lab (returns NaN if none).
    """
    idx = np.where((pol_sorted[:-1] == from_lab) & (pol_sorted[1:] == to_lab))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return 0.5 * (d_sorted[i] + d_sorted[i+1])


def extract_boundaries_time(D, Z1_grid, Z2_grid, tol=1e-4):
    """
    For each t:
      1) collapse policy -> pol(d)
      2) boundary_hi: Wait(1) -> Choose1(2)
         boundary_lo: Choose2(3) -> Wait(1)

    Returns upper, lower arrays (len=ts)
    """
    d_vals = (Z1_grid - Z2_grid) / 2.0
    ts = D.shape[2]
    upper = np.full(ts, np.nan)
    lower = np.full(ts, np.nan)

    for t in range(ts):
        d_s, pol_s = policy_along_d(D[:, :, t], d_vals, tol=tol)
        # lower boundary: Choose2 -> Wait
        lower[t] = find_boundary(d_s, pol_s, from_lab=3, to_lab=1)
        # upper boundary: Wait -> Choose1
        upper[t] = find_boundary(d_s, pol_s, from_lab=1, to_lab=2)

    return upper, lower


def plot_boundaries_vs_time(params, rhos,
                            solver=solve_gaussian_dp,
                            tol=1e-4,
                            colors=None,
                            ls=('-', '-')):
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(rhos)))

    t_axis = np.arange(params["ts"]) * params["dt"]
    z_max, zs = params["z_max"], params["zs"]
    Z1 = np.linspace(-z_max, z_max, zs)
    Z2 = np.linspace(-z_max, z_max, zs)
    Z1_grid, Z2_grid = np.meshgrid(Z1, Z2, indexing='xy')

    fig, ax = plt.subplots(figsize=(6,4))
    curves = {}

    for c, rho in zip(colors, rhos):
        p = deepcopy(params); p["rho"] = rho
        V, D = solver(p)

        upper, lower = extract_boundaries_time(D, Z1_grid, Z2_grid, tol=tol)
        curves[rho] = (upper, lower)

        ax.plot(t_axis, upper, color=c, linestyle=ls[0],
                label=fr'$\rho={rho}$: Wait→C1')
        ax.plot(t_axis, lower, color=c, linestyle=ls[1],
                label=fr'$\rho={rho}$: C2→Wait')

    ax.set_xlabel('Time')
    ax.set_ylabel(r'$(\hat r_1 - \hat r_2)/2$')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('Policy boundaries over time')
    ax.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    return fig, ax, curves


if __name__ == "__main__":
    params = get_params(which_tree=1)

    # V, D = solve_gaussian_dp(params)

    z_max = params["z_max"]
    zs = params["zs"]
    ts = params["ts"]

    R1 = np.linspace(-z_max, z_max, zs)
    R2 = np.linspace(-z_max, z_max, zs)

    R1_grid, R2_grid = np.meshgrid(R1, R2)

    # plot_value_function(V, R1_grid, R2_grid, t=0)
    # plot_decision_function(D, R1_grid, R2_grid, t=0)

    # ani_V = animate_value_function(V, R1_grid, R2_grid)
    # ani_D = animate_policy(D, R1_grid, R2_grid)

    # # save the animation
    # # ani_V.save('value_function.mp4', writer='ffmpeg', fps=10)
    # ani_D.save('policy_1.gif', writer='imagemagick', fps=10)

    rhos = [-0.5, -0.2, 0.0, 0.2, 0.5]
    fig, ax, curves = plot_boundaries_vs_time(params, rhos)
    plt.show()
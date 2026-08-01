import numpy as np
from scipy.ndimage import convolve1d
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
from getParams import get_params
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from IPython.display import HTML
from IPython.display import display

import os

def solve_3d(params):

    zs, ts, z_max, dt, rho, base_node, utility, cost, t_null = (
        params["zs"], params["ts"], params["z_max"], params["dt"], params["rho"], params["base_node"], params["utility"], params["cost"], params["t_null2"]
    )

    Z1 = np.linspace(-z_max, z_max, zs)
    Z2 = np.linspace(-z_max, z_max, zs)
    Z3 = np.linspace(-z_max, z_max, zs)
    Z1_grid, Z2_grid, Z3_grid = np.meshgrid(Z1, Z2, Z3, indexing="ij")

    V = np.empty((zs, zs, zs, ts))
    D = np.empty((zs, zs, zs, ts))

    Rh = [
        utility(Z1_grid), 
        utility(Z2_grid), 
        utility(Z3_grid),
    ]

    V = np.empty((zs, zs, zs, ts))
    D = np.empty((zs, zs, zs, ts), dtype=np.uint8)

    term_Q = np.stack([Rh[i] - rho * t_null for i in range(3)], axis=3)  # (zs,zs,zs,3)
    V[..., ts-1] = np.max(term_Q, axis=3)
    D[..., ts-1] = np.argmax(term_Q, axis=3) + 1  # 1,2,3 分别对应三种决策

    for t in tqdm(range(ts-2, -1, -1), desc="Stage-3 DP"):
        time = t * dt

        # 3D convolution kernel (update posterior variance for each dimension)
        var1 = var_trans(base_node, time, dt)
        var2 = var_trans(base_node, time, dt)
        var3 = var_trans(base_node, time, dt)
        kernel = sep_kernel_3d(-z_max, z_max, zs, (var1, var2, var3))

        # expected value of waiting action
        EVnext = ndimage.convolve(V[..., t+1], kernel, mode="nearest")
        wait = EVnext - (rho + cost) * dt

        # three immediate choices + waiting, take max
        choose_Q = [Rh[i] - rho * t_null for i in range(3)]
        Q = np.stack([*choose_Q, wait], axis=3)  # 4 actions

        V[..., t] = np.max(Q, axis=3)
        D[..., t] = np.argmax(Q, axis=3) + 1     # 1=choose1,2=choose2,3=choose3,4=wait

    return V, D

def solve_3d_fast(params):

    '''
    This function is used to solve the 3D optimal decision problem.
    It uses the fast convolution algorithm to solve the problem.
    (f * K_1) * K_2 * K_3 = f * (K_1 \otimes K_2 \otimes K_3)
    '''

    zs, ts, z_max, dt, rho, base_node, utility, cost, t_null = (
        params["zs"], params["ts"], params["z_max"],
        params["dt"], params["rho"], params["base_node"],
        params["utility"], params["cost"], params["t_null2"]
    )

    coords = np.linspace(-z_max, z_max, zs)
    Z1, Z2, Z3 = np.meshgrid(coords, coords, coords, indexing="ij")

    Rh = [utility(Z1), utility(Z2), utility(Z3)]

    kernels = np.zeros((ts, zs))
    for ti in range(ts):
        var = var_trans(base_node, ti*dt, dt)
        K = np.exp(-coords**2 / (2 * var))
        kernels[ti] = K / K.sum()

    # Value function and decision function
    V = np.empty((zs, zs, zs, ts), dtype=np.float32)
    D = np.empty((zs, zs, zs, ts), dtype=np.uint8)

    # terminal
    term_Q = np.stack([Rh[i] - rho * t_null for i in range(3)], axis=3)
    V[..., ts-1] = term_Q.max(axis=3)
    D[..., ts-1] = term_Q.argmax(axis=3) + 1

    for t in tqdm(range(ts-2, -1, -1), desc="Multi-alternative Choice"):
        
        arr = V[..., t+1]
        K = kernels[t]
        arr = convolve1d(arr, K, axis=0, mode="nearest")
        arr = convolve1d(arr, K, axis=1, mode="nearest")
        EVnext = convolve1d(arr, K, axis=2, mode="nearest")
        
        
        wait = EVnext - (rho + cost) * dt
        choose_Q = [Rh[i] - rho * t_null for i in range(3)]

        # action value
        Q = np.stack([*choose_Q, wait], axis=3)

        V[..., t] = Q.max(axis=3)
        D[..., t] = Q.argmax(axis=3) + 1

    return V, D

# auxiliary functions
def var_trans(p, t, dt):
    pv = p.post_var(t)
    fac = (p.var_r / (p.var_r * (t+dt) + p.var_x))**2
    return fac * (p.var_x + pv * dt) * dt

def sep_kernel_3d(z_min, z_max, zs, var_tuple):
    """
    生成 separable 3D 卷积核，var_tuple=(var1,var2,var3)
    这里示例直接生成三维 Gaussian 核：每轴独立
    """
    v1, v2, v3 = var_tuple
    coords = np.linspace(z_min, z_max, zs)
    K1 = np.exp(-coords**2 / (2 * v1))
    K2 = np.exp(-coords**2 / (2 * v2))
    K3 = np.exp(-coords**2 / (2 * v3))
    K1 /= K1.sum(); K2 /= K2.sum(); K3 /= K3.sum()
    # outer product to get separable kernel
    return np.einsum("i,j,k->ijk", K1, K2, K3)


def animate_value_3d(V, coords, times, step=1, s=2, save_path=None, fps=5):
    """
    3D 动画展示价值函数 V(z1,z2,z3,t) 随时间变化。
    
    参数
    ----
    V         : ndarray, shape (zs, zs, zs, ts)
    coords    : 1D array, 长度 zs，对应 z1,z2,z3 坐标
    times     : 1D array, 长度 ts
    step      : int, 每帧采样步长，降低点数
    s         : int, 散点大小
    save_path : str, 保存路径，如 'value_animation.gif' 或 'value_animation.mp4'
    fps       : int, 帧率
    """
    zs, _, _, ts = V.shape

    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    Xf = X.ravel()[::step]
    Yf = Y.ravel()[::step]
    Zf = Z.ravel()[::step]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    vmin, vmax = V.min(), V.max()
    
    scat = ax.scatter([], [], [], c=[], cmap='viridis', vmin=vmin, vmax=vmax, s=s)
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('z3')
    
    def update(frame):
        ax.cla()
        val = V[:, :, :, frame].ravel()[::step]
        scat = ax.scatter(Xf, Yf, Zf, c=val, cmap='viridis', vmin=vmin, vmax=vmax, s=s)
        ax.set_title(f"Value Function at t = {times[frame]:.2f}s")
        ax.set_xlim(coords[0], coords[-1])
        ax.set_ylim(coords[0], coords[-1])
        ax.set_zlim(coords[0], coords[-1])
        ax.set_xlabel('z1')
        ax.set_ylabel('z2')
        ax.set_zlabel('z3')
        return scat,
    
    ani = FuncAnimation(fig, update, frames=ts, interval=1000//fps, blit=False)
    
    # 保存动画
    if save_path:
        print(f"保存价值函数动画到: {save_path}")
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps)
        else:
            ani.save(save_path + '.gif', writer='pillow', fps=fps)
        print("动画保存完成!")
    
    return HTML(ani.to_jshtml())

def animate_policy_3d(D, coords, times, step=1, s=5, save_path=None, fps=5):
    """
    3D 动画展示策略 D(z1,z2,z3,t) 随时间变化。
    
    参数
    ----
    D         : ndarray, shape (zs, zs, zs, ts)
    coords    : 1D array, 长度 zs，对应 z1,z2,z3 坐标
    times     : 1D array, 长度 ts
    step      : int, 每帧采样步长
    s         : int, 散点大小
    save_path : str, 保存路径，如 'policy_animation.gif' 或 'policy_animation.mp4'
    fps       : int, 帧率
    """
    zs, _, _, ts = D.shape
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    Xf = X.ravel()[::step]
    Yf = Y.ravel()[::step]
    Zf = Z.ravel()[::step]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义决策颜色和标签
    decision_colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}
    decision_labels = {1: 'Choose Option 1', 2: 'Choose Option 2', 3: 'Choose Option 3', 4: 'Wait'}
    
    # 创建图例 - 修复颜色问题
    legend_elements = []
    for decision, color in decision_colors.items():
        legend_elements.append(mpatches.Patch(color=color, label=decision_labels[decision]))
    
    def update(frame):
        ax.cla()
        dec = D[:, :, :, frame].ravel()[::step]
        
        # 为每个决策类型绘制散点
        for decision in [1, 2, 3, 4]:
            mask = dec == decision
            if np.any(mask):
                ax.scatter(Xf[mask], Yf[mask], Zf[mask], 
                          c=decision_colors[decision], s=s, alpha=0.7,
                          label=decision_labels[decision])
        
        ax.set_title(f"Policy Function at t = {times[frame]:.2f}s")
        ax.set_xlim(coords[0], coords[-1])
        ax.set_ylim(coords[0], coords[-1])
        ax.set_zlim(coords[0], coords[-1])
        ax.set_xlabel('z1')
        ax.set_ylabel('z2')
        ax.set_zlabel('z3')
        ax.legend(handles=legend_elements, loc='upper right')
        return ax,
    
    ani = FuncAnimation(fig, update, frames=ts, interval=1000//fps, blit=False)
    
    # 保存动画
    if save_path:
        print(f"保存策略函数动画到: {save_path}")
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps)
        else:
            ani.save(save_path + '.gif', writer='pillow', fps=fps)
        print("动画保存完成!")
    
    return HTML(ani.to_jshtml())

def animate_value_diag_plane(V, coords, times, save_path=None, fps=5):
    """
    Project V(z1,z2,z3,t) onto the plane orthogonal to (1,1,1), and animate it over time.
    """

    # Project basis
    u  = np.array([1,1,1],dtype=float)/np.sqrt(3)
    b1 = np.array([1,-1,0],dtype=float)/np.sqrt(2)
    b2 = np.cross(u, b1)

    # Pre-compute (x',y') coordinates for each grid point
    zs = coords.size
    ZZ = np.stack(np.meshgrid(coords,coords,coords,indexing='ij'),axis=-1)
    # Remove diagonal components
    parallel = (ZZ @ u)[...,None] * u[None,None,None,:]
    perp     = ZZ - parallel
    Xp = perp @ b1
    Yp = perp @ b2

    fig, ax = plt.subplots(figsize=(6,6))
    scat = ax.scatter([], [], c=[], s=20, cmap='viridis')
    cb   = plt.colorbar(scat, ax=ax)
    cb.set_label('Value')

    def update(ti):
        ax.clear()
        vals = V[...,ti].ravel()
        xpr  = Xp.ravel()
        ypr  = Yp.ravel()
        im = ax.scatter(xpr, ypr, c=vals, s=12, cmap='viridis', marker='s')
        ax.set_title(f"Value Projection at t={times[ti]:.2f}s")
        ax.set_xlabel('Proj along b1')
        ax.set_ylabel('Proj along b2')
        return im,

    ani = FuncAnimation(fig, update, frames=len(times),
                        interval=1000//fps, blit=False)
    if save_path:
        writer = 'pillow' if save_path.endswith('.gif') else 'ffmpeg'
        ani.save(save_path, writer=writer, fps=fps)
    return HTML(ani.to_jshtml())


def animate_policy_diag_plane(D, coords, times, save_path=None, fps=5):
    """
    Project D(z1,z2,z3,t) onto the plane orthogonal to (1,1,1), and animate it over time.
    Use four colors to distinguish four actions.
    """

    u  = np.array([1, 1, 1],dtype=float)/np.sqrt(3)
    b1 = np.array([1,-1, 0],dtype=float)/np.sqrt(2)
    b2 = np.cross(u, b1)

    zs = coords.size
    ZZ = np.stack(np.meshgrid(coords,coords,coords,indexing='ij'),axis=-1)
    parallel = (ZZ @ u)[...,None] * u[None,None,None,:]
    perp     = ZZ - parallel
    Xp = perp @ b1
    Yp = perp @ b2

    # 决策颜色映射
    cmap = {1:'red', 2:'blue', 3:'green', 4:'orange'}
    labels = {1:'Opt1',2:'Opt2',3:'Opt3',4:'Wait'}

    fig, ax = plt.subplots(figsize=(6,6))
    def update(ti):
        ax.clear()
        Di = D[...,ti].ravel()
        xpr=Xp.ravel(); ypr=Yp.ravel()
        for action in np.unique(Di):
            mask = (Di==action)
            ax.scatter(xpr[mask], ypr[mask],
                       c=cmap[action], s=12,
                       label=labels[action], alpha=0.6)
        ax.set_title(f"Policy Projection at t={times[ti]:.2f}s")
        ax.set_xlabel('Proj along b1')
        ax.set_ylabel('Proj along b2')
        ax.legend(loc='upper right', fontsize='small')
        return ax,

    ani = FuncAnimation(fig, update, frames=len(times),
                        interval=1000//fps, blit=False)
    if save_path:
        writer = 'pillow' if save_path.endswith('.gif') else 'ffmpeg'
        ani.save(save_path, writer=writer, fps=fps)
    
    return HTML(ani.to_jshtml())


if __name__ == "__main__":

    output_dir = "animations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    params = get_params(which_tree=1)
    V, D = solve_3d_fast(params)

    zs, ts, z_max, dt, rho, base_node, utility, cost, t_null = (
        params["zs"], params["ts"], params["z_max"],
        params["dt"], params["rho"], params["base_node"],
        params["utility"], params["cost"], params["t_null2"]
    )

    coords = np.linspace(-z_max, z_max, zs)
    times = np.arange(ts) * dt
    
    print("\n=== Generate animations ===")
    
    # # 1. 3D 价值函数动画
    # print("1. 生成3D价值函数动画...")
    # anim1 = animate_value_3d(V, coords, times, step=2, s=3, 
    #                        save_path=f"{output_dir}/value_function_3d.gif", fps=5)
    
    # # 2. 3D 策略函数动画
    # print("2. 生成3D策略函数动画...")
    # anim2 = animate_policy_3d(D, coords, times, step=2, s=5, 
    #                          save_path=f"{output_dir}/policy_function_3d.gif", fps=5)
    
    animate_value_diag_plane(V, coords, times, save_path='value_plane.gif', fps=5)
    animate_policy_diag_plane(D, coords, times, save_path='policy_plane.gif', fps=5)
    
    print(f"\n=== Animations saved to {output_dir}/ ===")

    



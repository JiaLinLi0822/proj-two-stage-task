import numpy as np
from tqdm import tqdm
from util import var_trans, sep_kernel
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from getParams import get_params
from matplotlib.colors import ListedColormap, BoundaryNorm

def solve_2d(params):

    zs, ts, z_max, dt, rho, base_node, utility, cost, t_null = (
        params["zs"], params["ts"], params["z_max"], params["dt"], 
        params["rho"], params["base_node"], params["utility"], 
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


# Example usage (once V, D, R1_grid, R2_grid are defined):
if __name__ == "__main__":
    params = get_params(which_tree=1)

    V, D = solve_2d(params)

    z_max = params["z_max"]
    zs = params["zs"]
    ts = params["ts"]

    R1 = np.linspace(-z_max, z_max, zs)
    R2 = np.linspace(-z_max, z_max, zs)
    R1_grid, R2_grid = np.meshgrid(R1, R2)
    
    # plot_value_function(V, R1_grid, R2_grid, t=0)
    # plot_decision_function(D, R1_grid, R2_grid, t=0)

    ani_V = animate_value_function(V, R1_grid, R2_grid)
    ani_D = animate_policy(D, R1_grid, R2_grid)

    # save the animation
    # ani_V.save('value_function.mp4', writer='ffmpeg', fps=5)
    # ani_D.save('policy.gif', writer='ffmpeg', fps=10)
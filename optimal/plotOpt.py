import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from getParams import get_params
from main import solve_leaf

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
    
    # Create initial contour plot
    cf = ax.contourf(
        R1_grid, R2_grid, D[:, :, 0],
        levels=[0.5, 1.5, 2.5, 3.5],
        cmap='viridis'
    )
    plt.colorbar(cf, ax=ax, label='Decision')
    ax.set_title('Policy at t = 0')

    def update(frame):
        # Clear previous contour
        for coll in ax.collections:
            coll.remove()
        
        # Create new contour
        cf = ax.contourf(
            R1_grid, R2_grid, D[:, :, frame],
            levels=[0.5, 1.5, 2.5, 3.5],
            cmap='viridis'
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

    V, D = solve_leaf(params)

    z_max = params["z_max"]
    zs = params["zs"]
    ts = params["ts"]

    R1 = np.linspace(-z_max, z_max, zs)
    R2 = np.linspace(-z_max, z_max, zs)
    R1_grid, R2_grid = np.meshgrid(R1, R2)
    
    plot_value_function(V, R1_grid, R2_grid, t=0)
    plot_decision_function(D, R1_grid, R2_grid, t=0)

    ani_V = animate_value_function(V, R1_grid, R2_grid)
    ani_D = animate_policy(D, R1_grid, R2_grid)
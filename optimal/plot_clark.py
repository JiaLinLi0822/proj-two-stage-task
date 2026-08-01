import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def expected_max_joint_gaussian(mu_x, mu_y, sigma_x=1.0, sigma_y=1.0, rho=0.0):
    mu_x = np.asarray(mu_x)
    mu_y = np.asarray(mu_y)
    var_sum = sigma_x**2 + sigma_y**2 - 2 * rho * sigma_x * sigma_y
    std = np.sqrt(var_sum)
    d = (mu_x - mu_y) / std

    E_max = mu_x * norm.cdf(d) + mu_y * norm.cdf(-d) + std * norm.pdf(d)
    return E_max

# Grid of means
mu_range = np.linspace(-3, 3, 100)
MU_X, MU_Y = np.meshgrid(mu_range, mu_range)
Z = expected_max_joint_gaussian(MU_X, MU_Y, sigma_x=1.0, sigma_y=1.0, rho=0.5)

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# viridis色图，半透明
ax.plot_surface(MU_X, MU_Y, Z, cmap='viridis', edgecolor='none', alpha=0.5)
ax.set_xlabel('μ_X')
ax.set_ylabel('μ_Y')
ax.set_zlabel('E[max(X, Y)]')
ax.set_title('Expected Maximum of Jointly Gaussian Variables')

# Add value contours to 3D plot (深红色)
contour_levels = [0, 1, 2]
contours = ax.contour(MU_X, MU_Y, Z, levels=contour_levels, zdir='z', offset=None, colors='darkred', linestyles='solid', linewidths=2)
ax.clabel(contours, fmt='%.1f', colors='darkred')

plt.tight_layout()
plt.show()

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.contourf(MU_X, MU_Y, Z, levels=100, cmap='viridis')
plt.colorbar(label='E[max(X, Y)]')
plt.xlabel('μ_X')
plt.ylabel('μ_Y')
plt.title('Heatmap of E[max(X, Y)] (Clark formula)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')

# Add value contours to heatmap
contours2 = plt.contour(MU_X, MU_Y, Z, levels=contour_levels, colors='k', linestyles='solid')
plt.clabel(contours2, fmt='%.1f', colors='k')

plt.tight_layout()
plt.show()
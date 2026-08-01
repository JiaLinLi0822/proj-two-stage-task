import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal  # use this

# -------------------------
# 1. Define a 3D Gaussian and the probability to compute
#    Target: p = P(X1>0, X2>0, X3>0)
# -------------------------
mean = np.array([0.2, -0.1, 0.3])
cov = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.4],
    [0.3, 0.4, 1.0]
])

lower = np.array([0.0, 0.0, 0.0])
upper = np.array([np.inf, np.inf, np.inf])

# -------------------------
# 2. Compute MC reference with N=2e6
# -------------------------
N_ref = int(2e6)
print(f"Computing MC reference with N={N_ref}...")
t0 = time.perf_counter()
samples_ref = np.random.multivariate_normal(mean, cov, size=N_ref)
hits_ref = np.all(samples_ref > 0.0, axis=1)
p_mc_ref = hits_ref.mean()
t_mc_ref = time.perf_counter() - t0
print(f"MC reference (N={N_ref}): p = {p_mc_ref:.8f}, time {t_mc_ref:.3f}s")

# -------------------------
# 3. Compute Genz estimate
# -------------------------
t0 = time.perf_counter()
dist_neg = multivariate_normal(mean=-mean, cov=cov)
p_genz = dist_neg.cdf(np.zeros(3))   # = P(-X1<=0, -X2<=0, -X3<=0) = P(X>0)
t_genz = time.perf_counter() - t0
genz_error = abs(p_genz - p_mc_ref)
print(f"Genz estimate: p = {p_genz:.8f}, error vs MC_ref = {genz_error:.8e}, time {t_genz*1000:.3f} ms")

# -------------------------
# 4. Monte Carlo estimation for different sample sizes
# -------------------------
N_list = [10**3, 3*10**3, 10**4, 3*10**4, 10**5, 3*10**5]
mc_errors = []
mc_times = []
mc_estimates = []

for N in N_list:
    t0 = time.perf_counter()
    samples = np.random.multivariate_normal(mean, cov, size=N)
    hits = np.all(samples > 0.0, axis=1)
    p_mc = hits.mean()
    t_mc = time.perf_counter() - t0

    mc_errors.append(abs(p_mc - p_mc_ref))
    mc_times.append(t_mc)
    mc_estimates.append(p_mc)

    print(f"N={N:7d}, MC estimate={p_mc:.6f}, error vs MC_ref={abs(p_mc-p_mc_ref):.6e}, time={t_mc:.3f}s")

# -------------------------
# 5. Plot: error vs N, time vs N, and mark Genz time
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# (a) Error vs sample size (compared to MC reference N=2e6)
ax = axes[0]
ax.loglog(N_list, mc_errors, marker="o", label="Monte Carlo error vs MC(2e6)")
# Add Genz error as a horizontal line
ax.axhline(genz_error, linestyle="--", color="red", label=f"Genz error vs MC(2e6) = {genz_error:.2e}")
ax.set_xlabel("Sample size N (log)")
ax.set_ylabel("Absolute error |estimate - MC(2e6)| (log)")
ax.set_title("Error vs MC reference (N=2e6)")
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend()

# (b) Time vs sample size, and plot Genz time
ax = axes[1]
ax.loglog(N_list, mc_times, marker="o", label="Monte Carlo time")
ax.axhline(t_genz, linestyle="--", label=f"Genz time ≈ {t_genz:.4f}s")
ax.set_xlabel("Sample size N (log)")
ax.set_ylabel("Time / seconds (log)")
ax.set_title("Monte Carlo time vs Genz")
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend()

plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    csv_paths = {
        "PDA": "parameter_recovery/parameter_recovery_model5_20251024_171048.csv",
        "RSS": "parameter_recovery/parameter_recovery_model5_rss_20251024_173503.csv",
        "IBS": "parameter_recovery/parameter_recovery_model5_ibs_20251025_035653.csv"
    }
    
    dfs = {}
    for method, path in csv_paths.items():
        dfs[method] = pd.read_csv(path)

    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.labelsize': 9,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8
    })

    fig, axes = plt.subplots(2, 3, figsize=(6, 4), dpi=300)

    params = [
        ("θ1", r"$\theta_1$"),
        ("Δ",  r"$\Delta$"),
    ]

    methods = ["PDA", "RSS", "IBS"]

    for row_idx, (pname, plabel) in enumerate(params):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            df = dfs[method]
            
            true_col = f"true_{pname}"
            fit_col = f"fitted_{pname}"

            x = df[true_col].values
            y = df[fit_col].values

            r = np.corrcoef(x, y)[0, 1]
            
            ax.plot([0, 1], [0, 1], linewidth=0.8, color='gray', linestyle='--', alpha=0.7)

            ax.scatter(x, y, s=10, alpha=0.8)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

            if row_idx == 0:
                ax.text(0.5, 1.05, method, transform=ax.transAxes, 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

            ax.set_xlabel(f"True {plabel}")
            ax.set_ylabel(f"Fitted {plabel}")

            ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

    plt.subplots_adjust(wspace=0.4, hspace=0.1, left=0.08, right=0.98, bottom=0.05, top=0.95)
    plt.savefig("parameter_recovery/parameter_recovery_model1.png")
    plt.show()


if __name__ == "__main__":
    main()
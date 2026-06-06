"""
=============================================================================
DATA ANALYSIS & VISUALIZATION MODULE
=============================================================================
Performs EDA (Exploratory Data Analysis) on the projectile motion dataset.

WHAT THIS MODULE DOES:
1. Data quality checks (missing values, dtypes, outliers)
2. Statistical summaries
3. Feature correlation analysis
4. Professional publication-quality visualizations
5. Saves all figures to /images/

WHY EDA MATTERS:
Before training any ML model, EDA reveals:
- Data distribution (normal? skewed? bimodal?)
- Feature correlations (which inputs most predict output?)
- Outliers that could distort model training
- Whether the data matches expected physics
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import os

# ─── PLOT STYLING ─────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]
FIGSIZE_WIDE  = (16, 6)
FIGSIZE_SQUARE = (10, 8)
FIGSIZE_TALL   = (14, 10)
DPI = 150

def setup_ax(ax, title, xlabel, ylabel, fontsize=12):
    ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    return ax


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
def load_data(data_dir: str):
    main_df  = pd.read_csv(os.path.join(data_dir, "projectile_data.csv"))
    traj_df  = pd.read_csv(os.path.join(data_dir, "trajectories.csv"))
    return main_df, traj_df


# ─── 1. DATA QUALITY REPORT ──────────────────────────────────────────────────
def data_quality_report(df: pd.DataFrame):
    print("=" * 65)
    print("DATA QUALITY REPORT")
    print("=" * 65)
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Duplicate rows : {df.duplicated().sum()}")
    print(f"\n  Column dtypes:")
    for col in df.columns:
        print(f"    {col:<18} {str(df[col].dtype):<12} "
              f"min={df[col].min():.3f}  max={df[col].max():.3f}")
    print()
    return df.describe()


# ─── 2. DISTRIBUTION PLOTS ───────────────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame, images_dir: str):
    features = ["v0", "theta_deg", "t_fraction", "h_max", "range", "t_flight"]
    labels   = ["Initial Velocity v₀ (m/s)", "Launch Angle θ (°)",
                "Time Fraction", "Max Height (m)", "Range (m)", "Time of Flight (s)"]

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_TALL)
    axes = axes.flatten()

    for i, (feat, label) in enumerate(zip(features, labels)):
        ax = axes[i]
        ax.hist(df[feat], bins=60, color=PALETTE[i], edgecolor="white",
                linewidth=0.4, alpha=0.9)
        ax.axvline(df[feat].mean(), color="#222222", linestyle="--",
                   linewidth=1.5, label=f"mean={df[feat].mean():.2f}")
        setup_ax(ax, f"Distribution of {label}", label, "Count")
        ax.legend(fontsize=9)

    fig.suptitle("Feature Distributions — Projectile Motion Dataset",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(images_dir, "01_feature_distributions.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── 3. CORRELATION MATRIX ────────────────────────────────────────────────────
def plot_correlation_matrix(df: pd.DataFrame, images_dir: str):
    cols = ["v0", "theta_deg", "v0x", "v0y", "t", "x_t", "y_t",
            "h_max", "range", "t_flight", "speed_t"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")

    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(images_dir, "02_correlation_matrix.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── 4. TRAJECTORY PLOTS ─────────────────────────────────────────────────────
def plot_trajectories(traj_df: pd.DataFrame, images_dir: str):
    """Plot a sample of projectile trajectories coloured by launch angle."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # ── Panel A: coloured by theta ──
    ax = axes[0]
    sample_ids = traj_df["traj_id"].unique()[:50]   # plot 50 trajectories
    cmap = cm.get_cmap("plasma", len(sample_ids))

    for i, tid in enumerate(sample_ids):
        sub = traj_df[traj_df["traj_id"] == tid]
        ax.plot(sub["x"], sub["y"], color=cmap(i), linewidth=1.2, alpha=0.75)

    sm = plt.cm.ScalarMappable(cmap="plasma")
    sm.set_array([traj_df[traj_df["traj_id"].isin(sample_ids)]["traj_id"]])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Trajectory Index", fontsize=9)
    setup_ax(ax, "Sample Projectile Trajectories\n(50 random paths)",
             "Horizontal Distance x (m)", "Vertical Height y (m)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ── Panel B: 45° is max range ──
    ax = axes[1]
    angles = [15, 30, 45, 60, 75]
    v0_fixed = 50.0
    colors = PALETTE[:5]

    for ang, col in zip(angles, colors):
        rad = np.deg2rad(ang)
        v0x = v0_fixed * np.cos(rad)
        v0y = v0_fixed * np.sin(rad)
        t_f = 2 * v0y / 9.81
        t_vals = np.linspace(0, t_f, 300)
        x_vals = v0x * t_vals
        y_vals = np.maximum(0, v0y * t_vals - 0.5 * 9.81 * t_vals**2)
        ax.plot(x_vals, y_vals, color=col, linewidth=2.2,
                label=f"θ = {ang}°  R={x_vals[-1]:.0f}m")

    ax.axvline(x=v0_fixed**2/9.81, color="gray", linestyle=":", linewidth=1,
               label="45° max range")
    setup_ax(ax, f"Effect of Launch Angle\n(v₀ = {v0_fixed} m/s, fixed)",
             "Horizontal Distance x (m)", "Vertical Height y (m)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.suptitle("Projectile Trajectory Visualizations", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(images_dir, "03_trajectories.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── 5. PHYSICS RELATIONSHIPS ────────────────────────────────────────────────
def plot_physics_relationships(df: pd.DataFrame, images_dir: str):
    """Verify that ML inputs follow expected physics laws."""
    sample = df.sample(2000, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Range vs theta (should peak at 45°)
    ax = axes[0, 0]
    sc = ax.scatter(sample["theta_deg"], sample["range"],
                    c=sample["v0"], cmap="viridis", s=8, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="v₀ (m/s)")
    setup_ax(ax, "Range vs Launch Angle\n(coloured by initial velocity)",
             "Launch Angle θ (°)", "Range R (m)")
    ax.axvline(45, color="red", linestyle="--", linewidth=1.5, label="45° (max range)")
    ax.legend(fontsize=9)

    # Max height vs v0y²
    ax = axes[0, 1]
    ax.scatter(sample["v0y"]**2, sample["h_max"],
               color=PALETTE[1], s=8, alpha=0.4)
    v0y_sorted = np.sort(sample["v0y"]**2)
    ax.plot(v0y_sorted, v0y_sorted / (2*9.81), color="red",
            linewidth=2, label="Physics: H = v₀y²/(2g)")
    setup_ax(ax, "Max Height vs v₀y²\n(Physics equation overlay)",
             "v₀y² (m²/s²)", "Max Height H (m)")
    ax.legend(fontsize=9)

    # Range vs v0² * sin(2θ)
    ax = axes[1, 0]
    physics_range = sample["v0"]**2 * np.sin(2 * sample["theta_rad"]) / 9.81
    ax.scatter(physics_range, sample["range"],
               color=PALETTE[2], s=8, alpha=0.4)
    lim = max(physics_range.max(), sample["range"].max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=2, label="Perfect fit (y=x)")
    setup_ax(ax, "Physics-Derived Range vs Actual Range\n(Sanity check: should be y=x)",
             "v₀² sin(2θ)/g (m)", "Range R (m)")
    ax.legend(fontsize=9)

    # Time of flight vs v0y
    ax = axes[1, 1]
    ax.scatter(sample["v0y"], sample["t_flight"],
               color=PALETTE[3], s=8, alpha=0.4)
    v0y_sorted_vals = np.sort(sample["v0y"])
    ax.plot(v0y_sorted_vals, 2*v0y_sorted_vals/9.81, color="red",
            linewidth=2, label="Physics: T = 2v₀y/g")
    setup_ax(ax, "Time of Flight vs v₀y\n(Physics equation overlay)",
             "v₀y (m/s)", "Time of Flight T (s)")
    ax.legend(fontsize=9)

    plt.suptitle("Physics Relationship Verification", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(images_dir, "04_physics_relationships.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── 6. SCATTER MATRIX (KEY FEATURES) ───────────────────────────────────────
def plot_scatter_matrix(df: pd.DataFrame, images_dir: str):
    cols = ["v0", "theta_deg", "x_t", "y_t", "h_max", "range"]
    labels = ["v₀ (m/s)", "θ (°)", "x(t) (m)", "y(t) (m)", "H_max (m)", "R (m)"]
    sample = df[cols].sample(1500, random_state=42)
    sample.columns = labels

    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(16, 14))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(sample.iloc[:, i], bins=35, color=PALETTE[i % len(PALETTE)],
                        edgecolor="white", linewidth=0.3)
            else:
                ax.scatter(sample.iloc[:, j], sample.iloc[:, i],
                           s=2, alpha=0.3, color=PALETTE[i % len(PALETTE)])
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
            ax.tick_params(labelsize=6)

    plt.suptitle("Scatter Matrix — Key Projectile Motion Features",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(images_dir, "05_scatter_matrix.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE    = os.path.dirname(os.path.abspath(__file__)) + "/.."
    DATA    = os.path.join(BASE, "data")
    IMAGES  = os.path.join(BASE, "images")
    os.makedirs(IMAGES, exist_ok=True)

    print("=" * 65)
    print("RUNNING DATA ANALYSIS & VISUALIZATION")
    print("=" * 65)

    df, traj_df = load_data(DATA)

    stats = data_quality_report(df)
    print("\nDescriptive Statistics:\n")
    print(stats.to_string())

    print("\nGenerating plots...")
    plot_feature_distributions(df, IMAGES)
    plot_correlation_matrix(df, IMAGES)
    plot_trajectories(traj_df, IMAGES)
    plot_physics_relationships(df, IMAGES)
    plot_scatter_matrix(df, IMAGES)

    print("\n✓ All analysis plots saved to /images/")
"""
=============================================================================
MACHINE LEARNING MODELS — TRAINING & EVALUATION
=============================================================================
Trains multiple regression models to predict projectile positions (x, y)
from input features (v0, theta, t).

MODELS TRAINED:
1. Linear Regression      — Baseline; assumes linear feature relationships
2. Polynomial Regression  — Linear Regression on polynomial feature expansion
3. Random Forest          — Ensemble of decision trees; handles non-linearity

TARGETS:
  - x(t): horizontal position at time t
  - y(t): vertical position at time t

EVALUATION METRICS:
  MAE   = Mean Absolute Error           → average absolute deviation (same unit as target)
  MSE   = Mean Squared Error            → penalises large errors more heavily
  RMSE  = Root MSE                      → back in original units (interpretable)
  R²    = Coefficient of Determination  → 1.0 = perfect; 0 = predicting the mean

WHY MULTIPLE MODELS?
Each model has different inductive biases. Comparing them shows which class
of model best approximates the underlying physics.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── SETTINGS ─────────────────────────────────────────────────────────────────
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#44BBA4"]
DPI = 150
SEED = 42


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    if label:
        print(f"    {label}")
        print(f"      MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.6f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def build_pipelines():
    """
    Returns a dict of named sklearn pipelines.
    Each pipeline standardises features (important for Linear/Poly models)
    then applies its regressor.
    """
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ]),
        "PolynomialRegression_deg2": Pipeline([
            ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ]),
        "PolynomialRegression_deg3": Pipeline([
            ("poly",   PolynomialFeatures(degree=3, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ]),
        "RandomForest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=SEED
            ))
        ]),
    }


# ─── TRAINING ─────────────────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame):
    """
    Train all models, evaluate on hold-out test set, return results dict.

    FEATURE ENGINEERING:
    We expose the model to both raw inputs AND some physics-inspired
    transformations. The model should 'rediscover' that sin/cos of the
    angle matter, and that t² is important for y(t).
    """
    # ── Feature matrix ────────────────────────────────────────────────────────
    # Core inputs: v0, theta_deg (and its trig transforms), t
    # Also add: v0² (appears in range formula), sin(2θ) (max range)
    X = pd.DataFrame({
        "v0":          df["v0"],
        "v0_sq":       df["v0"] ** 2,
        "theta_deg":   df["theta_deg"],
        "sin_theta":   np.sin(df["theta_rad"]),
        "cos_theta":   np.cos(df["theta_rad"]),
        "sin2theta":   np.sin(2 * df["theta_rad"]),
        "t":           df["t"],
        "t_sq":        df["t"] ** 2,
        "v0_cos_t":    df["v0x"] * df["t"],          # ≈ x(t) by physics
        "v0_sin_t":    df["v0y"] * df["t"],           # part of y(t)
    })

    y_x = df["x_t"].values   # target 1: horizontal position
    y_y = df["y_t"].values   # target 2: vertical position

    # ── Train / test split (80/20) ────────────────────────────────────────────
    X_tr, X_te, yx_tr, yx_te, yy_tr, yy_te = train_test_split(
        X, y_x, y_y, test_size=0.2, random_state=SEED
    )

    print(f"\n  Training set : {len(X_tr):,} samples")
    print(f"  Test set     : {len(X_te):,} samples")
    print(f"  Features     : {X.shape[1]}")

    pipelines = build_pipelines()
    results = {}
    trained_models = {}

    print("\n" + "─" * 60)
    print("  TRAINING & EVALUATION RESULTS")
    print("─" * 60)

    for name, pipe in pipelines.items():
        print(f"\n  [{name}]")

        # ── Fit on training data ──────────────────────────────────────────────
        # For multi-output, we use two separate pipelines sharing the same
        # structure (sklearn does not multi-output Polynomial pipelines easily)
        import copy
        pipe_y = copy.deepcopy(pipe)

        pipe.fit(X_tr, yx_tr)
        pipe_y.fit(X_tr, yy_tr)

        # ── Predict on test set ───────────────────────────────────────────────
        yx_pred = pipe.predict(X_te)
        yy_pred = pipe_y.predict(X_te)

        metrics_x = compute_metrics(yx_te, yx_pred, "→ x(t) prediction")
        metrics_y = compute_metrics(yy_te, yy_pred, "→ y(t) prediction")

        results[name] = {
            "x": metrics_x,
            "y": metrics_y,
            "yx_pred": yx_pred,
            "yy_pred": yy_pred,
            "yx_te": yx_te,
            "yy_te": yy_te,
        }
        trained_models[name] = {"x": pipe, "y": pipe_y}

    return results, trained_models, X_te, yx_te, yy_te


# ─── VISUALIZATIONS ───────────────────────────────────────────────────────────
def plot_predicted_vs_actual(results: dict, images_dir: str):
    """4-panel predicted vs actual scatter plot for x(t) across models."""
    models = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, name in enumerate(models):
        ax = axes[i]
        yx_te   = results[name]["yx_te"]
        yx_pred = results[name]["yx_pred"]
        r2      = results[name]["x"]["R2"]
        rmse    = results[name]["x"]["RMSE"]

        ax.scatter(yx_te, yx_pred, s=6, alpha=0.3, color=PALETTE[i])
        lim = max(yx_te.max(), yx_pred.max())
        ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")

        short = name.replace("Regression", "Reg").replace("_", " ")
        ax.set_title(f"{short}\nR²={r2:.5f}   RMSE={rmse:.3f} m",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual x(t) (m)", fontsize=10)
        ax.set_ylabel("Predicted x(t) (m)", fontsize=10)
        ax.legend(fontsize=9)

    plt.suptitle("Predicted vs Actual — Horizontal Position x(t)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(images_dir, "06_predicted_vs_actual_x.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(results: dict, images_dir: str):
    """Bar chart comparing R², RMSE across models for both targets."""
    models   = list(results.keys())
    short    = [m.replace("Regression","Reg").replace("_"," ").replace("RandomForest","RF") for m in models]
    metrics  = ["R2", "RMSE", "MAE"]
    targets  = ["x", "y"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for k, metric in enumerate(metrics):
        ax = axes[k]
        x_vals = np.arange(len(models))
        w = 0.35

        bars_x = [results[m]["x"][metric] for m in models]
        bars_y = [results[m]["y"][metric] for m in models]

        b1 = ax.bar(x_vals - w/2, bars_x, width=w, color=PALETTE[0],
                    label="x(t)", alpha=0.85, edgecolor="white")
        b2 = ax.bar(x_vals + w/2, bars_y, width=w, color=PALETTE[1],
                    label="y(t)", alpha=0.85, edgecolor="white")

        ax.set_xticks(x_vals)
        ax.set_xticklabels(short, fontsize=9, rotation=20, ha="right")
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylabel(metric)

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=7)
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Model Comparison — All Metrics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(images_dir, "07_model_comparison.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_residuals(results: dict, images_dir: str):
    """Residual distribution plots for each model."""
    models = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, name in enumerate(models):
        ax = axes[i]
        residuals = results[name]["yx_te"] - results[name]["yx_pred"]
        ax.hist(residuals, bins=60, color=PALETTE[i], edgecolor="white",
                linewidth=0.3, alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
        ax.axvline(residuals.mean(), color="black", linestyle=":",
                   linewidth=1.5, label=f"Mean={residuals.mean():.2f}")
        short = name.replace("Regression", "Reg").replace("_", " ")
        ax.set_title(f"Residuals — {short}\n(σ={residuals.std():.3f} m)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Prediction Error (m)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)

    plt.suptitle("Residual Analysis — Horizontal Position x(t)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(images_dir, "08_residuals.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(trained_models: dict, feature_names: list, images_dir: str):
    """Feature importance from Random Forest model."""
    rf_pipe = trained_models["RandomForest"]["x"]
    rf_model = rf_pipe.named_steps["model"]
    importances = rf_model.feature_importances_

    sorted_idx = np.argsort(importances)
    sorted_feat = [feature_names[i] for i in sorted_idx]
    sorted_imp  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(sorted_feat)), sorted_imp,
                   color=PALETTE[0], edgecolor="white", alpha=0.9)
    ax.set_yticks(range(len(sorted_feat)))
    ax.set_yticklabels(sorted_feat, fontsize=11)
    ax.set_xlabel("Feature Importance (Gini Impurity Decrease)", fontsize=11)
    ax.set_title("Random Forest Feature Importance\n(for x(t) prediction)",
                 fontsize=13, fontweight="bold")
    for bar, val in zip(bars, sorted_imp):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(images_dir, "09_feature_importance.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_physics_vs_ml(trained_models: dict, images_dir: str):
    """
    Compare ML predictions against physics equations on a test trajectory.
    THIS IS THE KEY RESEARCH INSIGHT PLOT.
    """
    # Generate a test trajectory
    v0     = 50.0
    theta  = 45.0
    theta_r = np.deg2rad(theta)
    v0x    = v0 * np.cos(theta_r)
    v0y    = v0 * np.sin(theta_r)
    t_f    = 2 * v0y / 9.81
    t_vals = np.linspace(0, t_f, 100)

    # Physics ground truth
    x_phys = v0x * t_vals
    y_phys = np.maximum(0, v0y * t_vals - 0.5 * 9.81 * t_vals**2)

    # Build feature matrix for this trajectory
    X_test = pd.DataFrame({
        "v0":       [v0] * 100,
        "v0_sq":    [v0**2] * 100,
        "theta_deg": [theta] * 100,
        "sin_theta": [np.sin(theta_r)] * 100,
        "cos_theta": [np.cos(theta_r)] * 100,
        "sin2theta": [np.sin(2*theta_r)] * 100,
        "t":         t_vals,
        "t_sq":      t_vals**2,
        "v0_cos_t":  v0x * t_vals,
        "v0_sin_t":  v0y * t_vals,
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    model_names = ["LinearRegression", "PolynomialRegression_deg2",
                   "PolynomialRegression_deg3", "RandomForest"]
    colors = PALETTE[:4]

    for ax_idx, target in enumerate(["x", "y"]):
        ax = axes[ax_idx]
        truth = x_phys if target == "x" else y_phys
        ax.plot(t_vals, truth, "k-", linewidth=3, label="Physics (Ground Truth)", zorder=5)

        for name, col in zip(model_names, colors):
            pipe = trained_models[name][target]
            pred = pipe.predict(X_test)
            short = name.replace("Regression","Reg").replace("_"," ")
            ax.plot(t_vals, pred, "--", color=col, linewidth=1.8,
                    label=f"{short}", alpha=0.85)

        label = "x(t)" if target == "x" else "y(t)"
        ax.set_title(f"Physics vs ML Predictions — {label}\n(v₀=50 m/s, θ=45°)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Time t (s)", fontsize=11)
        ax.set_ylabel(f"{label} (m)", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.suptitle("Can ML Rediscover Physics? — Trajectory Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(images_dir, "10_physics_vs_ml.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────
def save_results(results: dict, trained_models: dict, results_dir: str, models_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Save metrics
    metrics_out = {}
    for name, res in results.items():
        metrics_out[name] = {"x": res["x"], "y": res["y"]}
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Saved metrics → {results_dir}/metrics.json")

    # Save trained models
    for name, pipes in trained_models.items():
        for target, pipe in pipes.items():
            fname = os.path.join(models_dir, f"{name}_{target}.pkl")
            joblib.dump(pipe, fname)
    print(f"  Saved models  → {models_dir}/")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE    = os.path.dirname(os.path.abspath(__file__)) + "/.."
    DATA    = os.path.join(BASE, "data")
    IMAGES  = os.path.join(BASE, "images")
    MODELS  = os.path.join(BASE, "models")
    RESULTS = os.path.join(BASE, "results")
    os.makedirs(IMAGES, exist_ok=True)

    print("=" * 60)
    print("MACHINE LEARNING TRAINING & EVALUATION")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA, "projectile_data.csv"))

    results, trained_models, X_te, yx_te, yy_te = train_and_evaluate(df)

    feature_names = ["v0", "v0_sq", "theta_deg", "sin_theta", "cos_theta",
                     "sin2theta", "t", "t_sq", "v0·cos·t", "v0·sin·t"]

    print("\nGenerating ML visualizations...")
    plot_predicted_vs_actual(results, IMAGES)
    plot_model_comparison(results, IMAGES)
    plot_residuals(results, IMAGES)
    plot_feature_importance(trained_models, feature_names, IMAGES)
    plot_physics_vs_ml(trained_models, IMAGES)

    save_results(results, trained_models, RESULTS, MODELS)

    print("\n✓ ML training & evaluation complete!")
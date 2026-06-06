"""
=============================================================================
PHYSICS DATASET GENERATOR
=============================================================================
Generates synthetic projectile motion data using classical physics equations.

PHYSICS BACKGROUND:
-------------------
Projectile motion is 2D kinematics under constant gravitational acceleration.
All air resistance is ignored (ideal projectile).

EQUATIONS USED:
---------------
Given:
    v0  = initial velocity (m/s)
    θ   = launch angle (degrees → radians)
    g   = gravitational acceleration = 9.81 m/s²

Derived:
    v0x = v0 * cos(θ)           → horizontal velocity component (constant)
    v0y = v0 * sin(θ)           → vertical velocity component (initial)

    x(t)    = v0x * t                        → horizontal position at time t
    y(t)    = v0y * t - 0.5 * g * t²        → vertical position at time t

    T_flight = 2 * v0y / g                  → total time of flight
    R        = v0² * sin(2θ) / g            → horizontal range
    H_max    = v0y² / (2 * g)               → maximum height

WHY SYNTHETIC DATA?
-------------------
Real projectile experiments have noise from air resistance, wind, spin, etc.
Synthetic data lets us test ML models against ground truth physics.
=============================================================================
"""

import numpy as np
import pandas as pd
import os

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
G = 9.81  # gravitational acceleration (m/s²)
SEED = 42  # reproducibility

def generate_projectile_dataset(n_samples: int = 10000, seed: int = SEED) -> pd.DataFrame:
    """
    Generate a synthetic projectile motion dataset.

    Parameters
    ----------
    n_samples : int
        Number of data points to generate (default 10,000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with features and targets for each projectile scenario.
    """
    rng = np.random.default_rng(seed)

    # ── INDEPENDENT VARIABLES (inputs to the physics system) ──────────────────
    # Initial velocity: uniform between 5 and 100 m/s
    # Real range: a sprint is ~10 m/s; a cannon ~500 m/s; 100 m/s is educational
    v0 = rng.uniform(5.0, 100.0, n_samples)

    # Launch angle: uniform between 1° and 89°
    # 0° and 90° are degenerate cases (no flight or purely vertical)
    theta_deg = rng.uniform(1.0, 89.0, n_samples)
    theta_rad = np.deg2rad(theta_deg)

    # ── VELOCITY COMPONENTS ───────────────────────────────────────────────────
    v0x = v0 * np.cos(theta_rad)   # Horizontal component (stays constant)
    v0y = v0 * np.sin(theta_rad)   # Vertical component (decreases due to gravity)

    # ── DERIVED PHYSICS QUANTITIES ────────────────────────────────────────────

    # Time of flight: when y(t) = 0 again → t = 2*v0y / g
    t_flight = 2.0 * v0y / G

    # Maximum height: apex at t = v0y/g → H = v0y²/(2g)
    h_max = (v0y ** 2) / (2.0 * G)

    # Horizontal range: x at t = t_flight → R = v0² * sin(2θ) / g
    range_ = (v0 ** 2) * np.sin(2.0 * theta_rad) / G

    # ── TIME-STEP SAMPLING ────────────────────────────────────────────────────
    # For each projectile, sample a random time t in [0, t_flight]
    # This gives us instantaneous position data (not just final range)
    t_fraction = rng.uniform(0.0, 1.0, n_samples)  # fraction of flight time
    t = t_fraction * t_flight

    # Position at time t
    x_t = v0x * t
    y_t = v0y * t - 0.5 * G * (t ** 2)

    # Velocity at time t (useful features for ML)
    vx_t = v0x                      # horizontal velocity is constant
    vy_t = v0y - G * t              # vertical velocity decreases linearly

    speed_t = np.sqrt(vx_t**2 + vy_t**2)   # total speed magnitude

    # Kinetic energy (proportional, mass=1 kg assumed)
    ke_t = 0.5 * speed_t ** 2

    # ── ADD GAUSSIAN NOISE to simulate sensor measurement error ───────────────
    # This makes the ML task more realistic and prevents overfitting to exact physics
    noise_scale = 0.001  # 0.1% noise
    x_t_noisy = x_t * (1 + rng.normal(0, noise_scale, n_samples))
    y_t_noisy = y_t * (1 + rng.normal(0, noise_scale, n_samples))
    y_t_noisy = np.maximum(y_t_noisy, 0)   # clamp: can't go below ground

    # ── ASSEMBLE DATAFRAME ────────────────────────────────────────────────────
    df = pd.DataFrame({
        # Input features
        "v0":           v0,
        "theta_deg":    theta_deg,
        "theta_rad":    theta_rad,
        "t":            t,
        "t_fraction":   t_fraction,

        # Velocity components (physics-derived features)
        "v0x":          v0x,
        "v0y":          v0y,
        "vx_t":         vx_t,
        "vy_t":         vy_t,
        "speed_t":      speed_t,
        "ke_t":         ke_t,

        # Target variables (what we want ML to predict)
        "x_t":          x_t,              # true horizontal position
        "y_t":          y_t,              # true vertical position
        "x_t_noisy":    x_t_noisy,        # measured (noisy) horizontal position
        "y_t_noisy":    y_t_noisy,        # measured (noisy) vertical position

        # Summary statistics per trajectory
        "t_flight":     t_flight,
        "h_max":        h_max,
        "range":        range_,
    })

    return df


def generate_full_trajectories(n_trajectories: int = 200, n_points: int = 50, seed: int = SEED) -> pd.DataFrame:
    """
    Generate full trajectory data for multiple projectiles.
    Each trajectory is sampled at n_points evenly spaced time steps.
    Used for visualization.
    """
    rng = np.random.default_rng(seed + 1)

    rows = []
    v0_vals = rng.uniform(10.0, 80.0, n_trajectories)
    theta_vals = rng.uniform(10.0, 80.0, n_trajectories)

    for traj_id, (v0, theta_deg) in enumerate(zip(v0_vals, theta_vals)):
        theta_rad = np.deg2rad(theta_deg)
        v0x = v0 * np.cos(theta_rad)
        v0y = v0 * np.sin(theta_rad)
        t_flight = 2.0 * v0y / G

        t_vals = np.linspace(0, t_flight, n_points)
        for t in t_vals:
            x = v0x * t
            y = max(0.0, v0y * t - 0.5 * G * t**2)
            rows.append({
                "traj_id": traj_id,
                "v0": v0,
                "theta_deg": theta_deg,
                "t": t,
                "x": x,
                "y": y,
                "t_flight": t_flight,
                "h_max": v0y**2 / (2*G),
                "range": v0**2 * np.sin(2*theta_rad) / G,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING PROJECTILE MOTION DATASET")
    print("=" * 60)

    # Generate main dataset
    df = generate_projectile_dataset(n_samples=10000)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    main_path = os.path.join(out_dir, "projectile_data.csv")
    df.to_csv(main_path, index=False)
    print(f"\n✓ Main dataset saved → {main_path}")
    print(f"  Shape: {df.shape}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    print(f"\nDescriptive Stats:\n{df.describe().to_string()}")

    # Generate trajectory data for visualization
    traj_df = generate_full_trajectories(n_trajectories=200)
    traj_path = os.path.join(out_dir, "trajectories.csv")
    traj_df.to_csv(traj_path, index=False)
    print(f"\n✓ Trajectory dataset saved → {traj_path}")
    print(f"  Shape: {traj_df.shape}")
    print("\nDataset generation complete!")
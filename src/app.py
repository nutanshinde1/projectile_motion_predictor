"""
=============================================================================
PHYSICS-INFORMED PROJECTILE MOTION PREDICTOR
Interactive Streamlit Dashboard
=============================================================================
Senior Developer Notes:
- This app sits on TOP of the existing ML pipeline (models/, src/, results/)
- It imports trained .pkl models and runs both physics equations AND ML models
- Zero retraining needed — models are loaded once at startup via @st.cache_resource
- All physics is computed in pure NumPy for speed
- Plotly is used for interactive charts (hover, zoom, pan)
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
# Makes src/ importable when app.py is run from the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

G = 9.81  # gravitational constant (m/s²)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
# Must be the FIRST streamlit call in the script
st.set_page_config(
    page_title="Projectile Motion Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
# Injects custom styles into the Streamlit app for a professional look.
# Streamlit has limited native styling; CSS injection is the standard approach.
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1627 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(56, 189, 248, 0.15);
    border-color: #38bdf8;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
    font-weight: 500;
}
.metric-unit {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 2px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f1f5f9;
    padding: 8px 0 4px 0;
    border-bottom: 2px solid #334155;
    margin-bottom: 16px;
    letter-spacing: 0.3px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #162032 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #e879f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin-top: 8px;
    font-weight: 400;
}

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    background: linear-gradient(90deg, #1e3a5f, #1e293b);
    border: 1px solid #38bdf8;
    color: #38bdf8;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-right: 6px;
}

/* ── Insight box ── */
.insight-box {
    background: linear-gradient(135deg, #0f2027, #1a2a3a);
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.92rem;
    color: #cbd5e1;
    line-height: 1.6;
}

/* ── Comparison table ── */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
    margin-top: 10px;
}
.compare-table th {
    background: #1e293b;
    color: #94a3b8;
    font-weight: 600;
    padding: 10px 14px;
    text-align: left;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 1px;
    border-bottom: 2px solid #334155;
}
.compare-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #1e293b;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
}
.compare-table tr:hover td { background: #1e293b44; }

/* ── Stmetric override ── */
[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 14px 0 !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(14,165,233,0.5) !important;
}

/* ── Slider ── */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# @st.cache_resource caches the models in memory for the app's lifetime.
# Without this, models reload on every user interaction — very slow for RF.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading trained ML models...")
def load_models():
    """
    Load all trained sklearn pipeline .pkl files.
    Returns a nested dict: models[name][target] = fitted pipeline.
    """
    model_names = [
        "LinearRegression",
        "PolynomialRegression_deg2",
        "PolynomialRegression_deg3",
        "RandomForest",
    ]
    loaded = {}
    for name in model_names:
        try:
            loaded[name] = {
                "x": joblib.load(os.path.join(MODELS_DIR, f"{name}_x.pkl")),
                "y": joblib.load(os.path.join(MODELS_DIR, f"{name}_y.pkl")),
            }
        except FileNotFoundError:
            st.warning(f"Model not found: {name}. Run src/models.py first.")

    return loaded


@st.cache_data
def load_metrics():
    """Load saved evaluation metrics from results/metrics.json."""
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS ENGINE
# Pure NumPy implementations of the classical kinematic equations.
# These are the GROUND TRUTH against which we compare ML predictions.
# ══════════════════════════════════════════════════════════════════════════════
def physics_summary(v0: float, theta_deg: float) -> dict:
    """
    Compute all key projectile motion quantities from initial conditions.

    Equations:
        v0x = v0 * cos(θ)
        v0y = v0 * sin(θ)
        T   = 2 * v0y / g          (time of flight)
        H   = v0y² / (2g)          (maximum height)
        R   = v0² * sin(2θ) / g    (horizontal range)
    """
    theta_rad = np.deg2rad(theta_deg)
    v0x = v0 * np.cos(theta_rad)
    v0y = v0 * np.sin(theta_rad)
    t_flight = 2.0 * v0y / G
    h_max    = (v0y ** 2) / (2.0 * G)
    range_   = (v0 ** 2) * np.sin(2.0 * theta_rad) / G
    return {
        "v0x": v0x, "v0y": v0y,
        "t_flight": t_flight,
        "h_max": h_max,
        "range": range_,
        "theta_rad": theta_rad,
    }


def physics_trajectory(v0: float, theta_deg: float, n_points: int = 300) -> tuple:
    """
    Generate full (x, y) trajectory arrays using classical equations.
    Returns (x_array, y_array, t_array).
    """
    p = physics_summary(v0, theta_deg)
    t = np.linspace(0, p["t_flight"], n_points)
    x = p["v0x"] * t
    y = np.maximum(0.0, p["v0y"] * t - 0.5 * G * t ** 2)
    return x, y, t


# ══════════════════════════════════════════════════════════════════════════════
# ML PREDICTION ENGINE
# Builds the same feature matrix used during training (src/models.py)
# and runs inference through the loaded sklearn pipelines.
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_matrix(v0: float, theta_deg: float, t_array: np.ndarray) -> pd.DataFrame:
    """
    Construct the exact feature DataFrame the models were trained on.
    CRITICAL: Feature names AND order must match training exactly.
    """
    theta_rad = np.deg2rad(theta_deg)
    v0x = v0 * np.cos(theta_rad)
    v0y = v0 * np.sin(theta_rad)
    n = len(t_array)

    return pd.DataFrame({
        "v0":        [v0]           * n,
        "v0_sq":     [v0 ** 2]      * n,
        "theta_deg": [theta_deg]    * n,
        "sin_theta": [np.sin(theta_rad)] * n,
        "cos_theta": [np.cos(theta_rad)] * n,
        "sin2theta": [np.sin(2 * theta_rad)] * n,
        "t":         t_array,
        "t_sq":      t_array ** 2,
        "v0_cos_t":  v0x * t_array,
        "v0_sin_t":  v0y * t_array,
    })


def ml_trajectory(v0: float, theta_deg: float, model_pipe_x, model_pipe_y,
                  n_points: int = 300) -> tuple:
    """
    Run ML inference to predict a full trajectory.
    Returns (x_pred, y_pred, t_array).
    """
    p = physics_summary(v0, theta_deg)
    t = np.linspace(0, p["t_flight"], n_points)
    X = build_feature_matrix(v0, theta_deg, t)
    x_pred = model_pipe_x.predict(X)
    y_pred = np.maximum(0.0, model_pipe_y.predict(X))
    return x_pred, y_pred, t


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHART BUILDERS
# Each function returns a go.Figure object that Streamlit renders with
# st.plotly_chart(). Plotly gives interactive hover/zoom/pan for free.
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "physics":  "#38bdf8",   # sky blue  → ground truth
    "linear":   "#a78bfa",   # violet    → Linear Regression
    "poly2":    "#34d399",   # emerald   → Poly Deg-2
    "poly3":    "#fbbf24",   # amber     → Poly Deg-3
    "rf":       "#f87171",   # red       → Random Forest
    "apex":     "#e879f9",   # pink      → special markers
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(15,23,42,0.0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Space Grotesk, sans-serif", color="#cbd5e1", size=12),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155", linecolor="#334155"),
    legend=dict(
        bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155",
        borderwidth=1, font=dict(size=11)
    ),
    margin=dict(l=50, r=30, t=50, b=50),
    hoverlabel=dict(bgcolor="#0f172a", bordercolor="#334155",
                    font=dict(family="JetBrains Mono", size=11)),
)


def chart_single_trajectory(v0, theta_deg, phys_x, phys_y, ml_x, ml_y,
                             phys_summary_data, model_label):
    """
    Main trajectory chart — physics (solid) vs ML (dashed).
    Includes apex marker, landing marker, and angle arc annotation.
    """
    fig = go.Figure()

    # ── Physics trajectory ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=phys_x, y=phys_y,
        mode="lines", name="Physics (Ground Truth)",
        line=dict(color=COLORS["physics"], width=3),
        hovertemplate="Physics<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
    ))

    # ── ML trajectory ──────────────────────────────────────────────────────
    ml_color = {
        "LinearRegression":           COLORS["linear"],
        "PolynomialRegression_deg2":  COLORS["poly2"],
        "PolynomialRegression_deg3":  COLORS["poly3"],
        "RandomForest":               COLORS["rf"],
    }.get(model_label, "#fff")

    short = model_label.replace("Regression", " Reg").replace("_", " ")
    fig.add_trace(go.Scatter(
        x=ml_x, y=ml_y,
        mode="lines", name=f"ML: {short}",
        line=dict(color=ml_color, width=2.5, dash="dash"),
        hovertemplate=f"ML ({short})<br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m<extra></extra>",
    ))

    # ── Apex marker ────────────────────────────────────────────────────────
    h_max  = phys_summary_data["h_max"]
    x_apex = phys_summary_data["v0x"] * (phys_summary_data["t_flight"] / 2)
    fig.add_trace(go.Scatter(
        x=[x_apex], y=[h_max],
        mode="markers+text",
        name="Apex",
        marker=dict(color=COLORS["apex"], size=12, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        text=[f"  H_max = {h_max:.1f} m"],
        textposition="middle right",
        textfont=dict(color=COLORS["apex"], size=11),
        hovertemplate=f"Apex<br>x={x_apex:.1f} m<br>H_max={h_max:.1f} m<extra></extra>",
    ))

    # ── Landing marker ─────────────────────────────────────────────────────
    r = phys_summary_data["range"]
    fig.add_trace(go.Scatter(
        x=[r], y=[0],
        mode="markers+text",
        name="Landing",
        marker=dict(color="#fb923c", size=12, symbol="x",
                    line=dict(color="white", width=2)),
        text=[f"  R = {r:.1f} m"],
        textposition="middle right",
        textfont=dict(color="#fb923c", size=11),
        hovertemplate=f"Landing<br>Range={r:.1f} m<extra></extra>",
    ))

    # ── Launch arrow annotation ────────────────────────────────────────────
    arrow_len = r * 0.12
    fig.add_annotation(
        x=arrow_len * np.cos(phys_summary_data["theta_rad"]),
        y=arrow_len * np.sin(phys_summary_data["theta_rad"]),
        ax=0, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=3, arrowsize=1.5, arrowwidth=2,
        arrowcolor="#38bdf8",
    )
    fig.add_annotation(
        x=arrow_len * 0.6 * np.cos(phys_summary_data["theta_rad"] / 2),
        y=arrow_len * 0.4 * np.sin(phys_summary_data["theta_rad"] / 2),
        text=f"θ = {theta_deg:.1f}°",
        showarrow=False,
        font=dict(color="#94a3b8", size=11),
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"Projectile Trajectory  |  v₀ = {v0} m/s  |  θ = {theta_deg}°",
            font=dict(size=15, color="#f1f5f9"), x=0.01
        ),
        xaxis_title="Horizontal Distance x (m)",
        yaxis_title="Vertical Height y (m)",
        height=450,
    )
    fig.update_yaxes(rangemode="nonnegative")
    fig.update_xaxes(rangemode="nonnegative")
    return fig


def chart_all_models_comparison(v0, theta_deg, models_dict):
    """
    Overlay all ML models + physics on one chart.
    Shows at a glance how well each model approximates ground truth.
    """
    fig = go.Figure()
    p = physics_summary(v0, theta_deg)
    t = np.linspace(0, p["t_flight"], 300)
    phys_x = p["v0x"] * t
    phys_y = np.maximum(0, p["v0y"] * t - 0.5 * G * t ** 2)

    fig.add_trace(go.Scatter(
        x=phys_x, y=phys_y,
        mode="lines", name="Physics",
        line=dict(color=COLORS["physics"], width=4),
    ))

    style_map = {
        "LinearRegression":           (COLORS["linear"], "dot",    "Linear Reg"),
        "PolynomialRegression_deg2":  (COLORS["poly2"],  "dash",   "Poly Deg-2"),
        "PolynomialRegression_deg3":  (COLORS["poly3"],  "dashdot","Poly Deg-3"),
        "RandomForest":               (COLORS["rf"],     "longdash","Random Forest"),
    }

    X = build_feature_matrix(v0, theta_deg, t)
    for name, pipes in models_dict.items():
        color, dash, label = style_map.get(name, ("#fff", "solid", name))
        x_pred = pipes["x"].predict(X)
        y_pred = np.maximum(0, pipes["y"].predict(X))
        fig.add_trace(go.Scatter(
            x=x_pred, y=y_pred,
            mode="lines", name=f"ML: {label}",
            line=dict(color=color, width=2, dash=dash),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="All Models vs Physics  |  Can ML Rediscover Physics?",
            font=dict(size=15, color="#f1f5f9"), x=0.01
        ),
        xaxis_title="Horizontal Distance x (m)",
        yaxis_title="Vertical Height y (m)",
        height=450,
    )
    fig.update_yaxes(rangemode="nonnegative")
    fig.update_xaxes(rangemode="nonnegative")
    return fig


def chart_velocity_components(v0, theta_deg):
    """
    Show vx(t) and vy(t) over time.
    vx is constant; vy decreases linearly — key physics insight.
    """
    p = physics_summary(v0, theta_deg)
    t = np.linspace(0, p["t_flight"], 300)
    vx = np.full_like(t, p["v0x"])
    vy = p["v0y"] - G * t
    speed = np.sqrt(vx**2 + vy**2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=vx, name="v_x (horizontal)", mode="lines",
                             line=dict(color=COLORS["physics"], width=2.5),
                             hovertemplate="t=%{x:.2f}s<br>vx=%{y:.2f} m/s<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=vy, name="v_y (vertical)", mode="lines",
                             line=dict(color=COLORS["apex"], width=2.5),
                             hovertemplate="t=%{x:.2f}s<br>vy=%{y:.2f} m/s<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=speed, name="|v| speed", mode="lines",
                             line=dict(color=COLORS["poly2"], width=2, dash="dot"),
                             hovertemplate="t=%{x:.2f}s<br>speed=%{y:.2f} m/s<extra></extra>"))
    fig.add_hline(y=0, line_dash="dot", line_color="#334155")
    fig.add_vline(x=p["t_flight"]/2, line_dash="dot", line_color="#334155",
                  annotation_text="Apex", annotation_font_color="#94a3b8")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Velocity Components Over Time",
                   font=dict(size=15, color="#f1f5f9"), x=0.01),
        xaxis_title="Time t (s)",
        yaxis_title="Velocity (m/s)",
        height=380,
    )
    return fig


def chart_metrics_radar(metrics_dict):
    """
    Radar (spider) chart comparing R² across all models for x and y targets.
    Gives an instant visual of which model performs best.
    """
    models  = list(metrics_dict.keys())
    r2_x    = [metrics_dict[m]["x"]["R2"] * 100 for m in models]
    r2_y    = [metrics_dict[m]["y"]["R2"] * 100 for m in models]
    labels  = [m.replace("Regression","Reg").replace("_"," ") for m in models]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r2_x + [r2_x[0]], theta=labels + [labels[0]],
        fill="toself", name="R² x(t)",
        line=dict(color=COLORS["physics"], width=2),
        fillcolor="rgba(56,189,248,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=r2_y + [r2_y[0]], theta=labels + [labels[0]],
        fill="toself", name="R² y(t)",
        line=dict(color=COLORS["apex"], width=2),
        fillcolor="rgba(232,121,249,0.15)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[98, 100.1],
                            gridcolor="#1e293b", linecolor="#334155",
                            tickcolor="#94a3b8", tickfont=dict(size=9)),
            angularaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
            bgcolor="rgba(15,23,42,0.6)",
        ),
        title=dict(text="Model R² Performance Radar",
                   font=dict(size=14, color="#f1f5f9"), x=0.01),
        height=380,
    )
    return fig


def chart_error_heatmap():
    """
    Heatmap of Range (R) across v0 × theta grid.
    Shows which parameter combinations produce the longest/shortest ranges.
    """
    v0_vals     = np.linspace(10, 100, 30)
    theta_vals  = np.linspace(5,  85,  30)
    R_grid      = np.zeros((len(theta_vals), len(v0_vals)))

    for i, th in enumerate(theta_vals):
        for j, v in enumerate(v0_vals):
            R_grid[i, j] = (v**2) * np.sin(2*np.deg2rad(th)) / G

    fig = go.Figure(go.Heatmap(
        z=R_grid,
        x=np.round(v0_vals, 1),
        y=np.round(theta_vals, 1),
        colorscale="Plasma",
        colorbar=dict(
    title=dict(text="Range (m)",font=dict(size=14))),
        hovertemplate="v₀=%{x} m/s<br>θ=%{y}°<br>R=%{z:.1f} m<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Horizontal Range Heatmap  |  v₀ × θ Grid",
                   font=dict(size=14, color="#f1f5f9"), x=0.01),
        xaxis_title="Initial Velocity v₀ (m/s)",
        yaxis_title="Launch Angle θ (°)",
        height=400,
    )
    return fig


def chart_predicted_vs_actual_scatter(v0, theta_deg, models_dict):
    """
    Scatter of ML predicted y vs physics y along a trajectory.
    Perfect model → all points on the diagonal y=x line.
    """
    p = physics_summary(v0, theta_deg)
    t = np.linspace(0, p["t_flight"], 200)
    y_true = np.maximum(0, p["v0y"] * t - 0.5 * G * t**2)
    X = build_feature_matrix(v0, theta_deg, t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_true, mode="lines", name="Perfect Fit (y=x)",
        line=dict(color="#334155", width=1.5, dash="dot"),
    ))

    style_map = {
        "LinearRegression":           (COLORS["linear"], "Linear Reg"),
        "PolynomialRegression_deg2":  (COLORS["poly2"],  "Poly Deg-2"),
        "PolynomialRegression_deg3":  (COLORS["poly3"],  "Poly Deg-3"),
        "RandomForest":               (COLORS["rf"],     "Random Forest"),
    }
    for name, pipes in models_dict.items():
        color, label = style_map.get(name, ("#fff", name))
        y_pred = np.maximum(0, pipes["y"].predict(X))
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode="markers", name=label,
            marker=dict(color=color, size=5, opacity=0.7),
            hovertemplate=f"{label}<br>Actual=%{{x:.2f}}<br>Predicted=%{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Predicted vs Actual y(t)  |  All Models",
                   font=dict(size=14, color="#f1f5f9"), x=0.01),
        xaxis_title="Actual y(t) (m)",
        yaxis_title="Predicted y(t) (m)",
        height=400,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPER: METRIC CARD
# ══════════════════════════════════════════════════════════════════════════════
def metric_card(label: str, value: str, unit: str = "", icon: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-unit">{unit}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load assets ────────────────────────────────────────────────────────
    models_dict = load_models()
    metrics     = load_metrics()

    # ══════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # All user inputs live in the sidebar. Streamlit reruns the entire
    # script top-to-bottom whenever any widget value changes.
    # ══════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px 0;'>
            <div style='font-size:2.5rem;'>🚀</div>
            <div style='font-size:1.1rem; font-weight:700; color:#f1f5f9;'>
                Projectile Predictor
            </div>
            <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>
                Physics × Machine Learning
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Launch Parameters")

        v0 = st.slider(
            "Initial Velocity v₀ (m/s)",
            min_value=5.0, max_value=100.0, value=50.0, step=0.5,
            help="Speed at which the projectile is launched"
        )

        theta_deg = st.slider(
            "Launch Angle θ (degrees)",
            min_value=1.0, max_value=89.0, value=45.0, step=0.5,
            help="Angle above the horizontal. 45° gives maximum range."
        )

        st.markdown("---")
        st.markdown("### 🤖 ML Model")
        selected_model = st.selectbox(
            "Select Model for Comparison",
            options=list(models_dict.keys()),
            format_func=lambda x: x.replace("Regression"," Reg").replace("_"," "),
            help="Which trained ML model to compare against physics"
        )

        st.markdown("---")

        predict_clicked = st.button("🚀 PREDICT", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.72rem; color:#475569; line-height:1.6;'>
        <b style='color:#64748b;'>Physics Equations Used:</b><br>
        x(t) = v₀cosθ · t<br>
        y(t) = v₀sinθ · t − ½gt²<br>
        H = v₀y² / 2g<br>
        R = v₀² sin(2θ) / g<br>
        T = 2v₀y / g<br><br>
        g = 9.81 m/s² · No air resistance
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # HERO BANNER
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🚀 Physics-Informed Projectile Motion Predictor</p>
        <p class="hero-subtitle">
            Comparing Classical Physics Equations with Machine Learning Models
            &nbsp;·&nbsp; 10,000 training samples &nbsp;·&nbsp; 4 ML models
            &nbsp;·&nbsp; Real-time interactive analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # COMPUTE (always runs, not just on button click)
    # This makes the dashboard live — sliders update charts instantly.
    # ══════════════════════════════════════════════════════════════════════
    p = physics_summary(v0, theta_deg)
    phys_x, phys_y, t_arr = physics_trajectory(v0, theta_deg)

    selected_pipes = models_dict.get(selected_model, None)
    ml_x, ml_y = phys_x, phys_y  # fallback
    if selected_pipes:
        ml_x, ml_y, _ = ml_trajectory(v0, theta_deg,
                                       selected_pipes["x"], selected_pipes["y"])

    # ══════════════════════════════════════════════════════════════════════
    # METRICS ROW — Physics summary cards
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📐 Physics Predictions</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Max Height",     f"{p['h_max']:.2f}",    "metres",    "🏔️")
    with c2: metric_card("Time of Flight", f"{p['t_flight']:.2f}", "seconds",   "⏱️")
    with c3: metric_card("Horizontal Range", f"{p['range']:.2f}",  "metres",    "📏")
    with c4: metric_card("v₀ Horizontal", f"{p['v0x']:.2f}",       "m/s",       "➡️")
    with c5: metric_card("v₀ Vertical",   f"{p['v0y']:.2f}",       "m/s",       "⬆️")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # MAIN TABS
    # ══════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Trajectory",
        "🤖 ML vs Physics",
        "📊 Analysis",
        "📈 Model Performance",
        "🔬 Research Insights",
    ])

    # ── TAB 1: TRAJECTORY ─────────────────────────────────────────────────
    with tab1:
        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.plotly_chart(
                chart_single_trajectory(v0, theta_deg, phys_x, phys_y,
                                        ml_x, ml_y, p, selected_model),
                use_container_width=True
            )
            st.plotly_chart(chart_velocity_components(v0, theta_deg),
                            use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">📋 Summary</div>',
                        unsafe_allow_html=True)

            # Physics vs ML comparison table
            if selected_pipes:
                mid_t = np.array([p["t_flight"] / 2])
                X_mid = build_feature_matrix(v0, theta_deg, mid_t)
                ml_x_mid = float(selected_pipes["x"].predict(X_mid)[0])
                ml_y_mid = float(selected_pipes["y"].predict(X_mid)[0])
                phys_x_mid = float(p["v0x"] * mid_t[0])
                phys_y_mid = float(p["h_max"])  # apex is at t_flight/2

                model_short = selected_model.replace("Regression", " Reg").replace("_", " ")
                st.markdown(f"""
                <table class="compare-table">
                    <tr>
                        <th>Metric</th>
                        <th>Physics</th>
                        <th>ML ({model_short})</th>
                    </tr>
                    <tr>
                        <td>x at apex</td>
                        <td>{phys_x_mid:.2f} m</td>
                        <td>{ml_x_mid:.2f} m</td>
                    </tr>
                    <tr>
                        <td>y at apex</td>
                        <td>{phys_y_mid:.2f} m</td>
                        <td>{ml_y_mid:.2f} m</td>
                    </tr>
                    <tr>
                        <td>Max Height</td>
                        <td>{p['h_max']:.2f} m</td>
                        <td>—</td>
                    </tr>
                    <tr>
                        <td>Range</td>
                        <td>{p['range']:.2f} m</td>
                        <td>—</td>
                    </tr>
                    <tr>
                        <td>Time of Flight</td>
                        <td>{p['t_flight']:.3f} s</td>
                        <td>—</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔑 Key Physics</div>',
                        unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-box">
            🔵 <b>Horizontal</b>: Constant velocity<br>
            &nbsp;&nbsp;&nbsp;x(t) = {p['v0x']:.2f} × t
            </div>
            <div class="insight-box">
            🟣 <b>Vertical</b>: Decelerates due to gravity<br>
            &nbsp;&nbsp;&nbsp;y(t) = {p['v0y']:.2f}t − 4.905t²
            </div>
            <div class="insight-box">
            {'✅ 45° launch = maximum range' if 43 <= theta_deg <= 47
             else f'📌 Range is {((p["range"] / ((v0**2)/G))*100):.1f}% of max possible.<br>Try θ=45° for maximum range.'}
            </div>
            """, unsafe_allow_html=True)

            # Angle insight
            if theta_deg < 45:
                comp = 90 - theta_deg
                st.markdown(f"""
                <div class="insight-box">
                💡 θ = {theta_deg}° and θ = {comp:.1f}° give
                the <b>same range</b> ({p['range']:.1f} m).
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 2: ML VS PHYSICS ───────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">🤖 All ML Models vs Physics</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_all_models_comparison(v0, theta_deg, models_dict),
                        use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(chart_predicted_vs_actual_scatter(v0, theta_deg, models_dict),
                            use_container_width=True)
        with col_b:
            st.markdown('<div class="section-header">📊 Model Metrics Summary</div>',
                        unsafe_allow_html=True)
            if metrics:
                rows = []
                for m, res in metrics.items():
                    rows.append({
                        "Model": m.replace("Regression"," Reg").replace("_"," "),
                        "R² x(t)": f"{res['x']['R2']:.6f}",
                        "R² y(t)": f"{res['y']['R2']:.6f}",
                        "RMSE x":  f"{res['x']['RMSE']:.6f}",
                        "RMSE y":  f"{res['y']['RMSE']:.4f}",
                    })
                df_m = pd.DataFrame(rows)
                st.dataframe(df_m, use_container_width=True, hide_index=True)

                st.markdown("""
                <div class="insight-box">
                💡 <b>Why R² = 1.0 for Linear/Poly models?</b><br>
                We included <code>v₀cosθ·t</code> as a feature — which IS the x(t) equation.
                The model just learns its coefficient is 1.0.
                This demonstrates: <b>feature engineering = encoding physics priors</b>.
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: ANALYSIS ────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">🌐 Parameter Space Analysis</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_error_heatmap(), use_container_width=True)

        col_x, col_y = st.columns(2)

        with col_x:
            # Range vs angle for fixed v0
            theta_sweep = np.linspace(1, 89, 200)
            ranges = (v0**2) * np.sin(2 * np.deg2rad(theta_sweep)) / G
            fig_sweep = go.Figure()
            fig_sweep.add_trace(go.Scatter(
                x=theta_sweep, y=ranges, mode="lines",
                line=dict(color=COLORS["physics"], width=2.5),
                hovertemplate="θ=%{x:.1f}°<br>R=%{y:.1f} m<extra></extra>",
            ))
            fig_sweep.add_vline(x=theta_deg, line_dash="dot",
                                line_color=COLORS["apex"],
                                annotation_text=f"Your θ={theta_deg}°",
                                annotation_font_color=COLORS["apex"])
            fig_sweep.add_vline(x=45, line_dash="dot", line_color="#38bdf8",
                                annotation_text="Max Range (45°)",
                                annotation_font_color="#38bdf8")
            fig_sweep.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"Range vs Angle  |  v₀={v0} m/s",
                           font=dict(size=13, color="#f1f5f9"), x=0.01),
                xaxis_title="Launch Angle θ (°)",
                yaxis_title="Range (m)", height=350,
            )
            st.plotly_chart(fig_sweep, use_container_width=True)

        with col_y:
            # Max height vs initial velocity for fixed theta
            v0_sweep = np.linspace(5, 100, 200)
            h_sweep = (v0_sweep * np.sin(np.deg2rad(theta_deg)))**2 / (2*G)
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=v0_sweep, y=h_sweep, mode="lines",
                line=dict(color=COLORS["poly2"], width=2.5),
                hovertemplate="v₀=%{x:.1f} m/s<br>H=%{y:.1f} m<extra></extra>",
            ))
            fig_h.add_vline(x=v0, line_dash="dot", line_color=COLORS["apex"],
                            annotation_text=f"Your v₀={v0}",
                            annotation_font_color=COLORS["apex"])
            fig_h.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"Max Height vs v₀  |  θ={theta_deg}°",
                           font=dict(size=13, color="#f1f5f9"), x=0.01),
                xaxis_title="Initial Velocity v₀ (m/s)",
                yaxis_title="Max Height (m)", height=350,
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # ── TAB 4: MODEL PERFORMANCE ───────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">📈 Trained Model Performance</div>',
                    unsafe_allow_html=True)
        col_radar, col_bar = st.columns([1, 1])

        with col_radar:
            if metrics:
                st.plotly_chart(chart_metrics_radar(metrics),
                                use_container_width=True)

        with col_bar:
            if metrics:
                models_list = list(metrics.keys())
                short_names = [m.replace("Regression","Reg").replace("_"," ") for m in models_list]
                rmse_x = [metrics[m]["x"]["RMSE"] for m in models_list]
                rmse_y = [metrics[m]["y"]["RMSE"] for m in models_list]

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(name="RMSE x(t)", x=short_names, y=rmse_x,
                                         marker_color=COLORS["physics"], opacity=0.85))
                fig_bar.add_trace(go.Bar(name="RMSE y(t)", x=short_names, y=rmse_y,
                                         marker_color=COLORS["apex"], opacity=0.85))
                fig_bar.update_layout(
                    **PLOTLY_LAYOUT,
                    barmode="group",
                    title=dict(text="RMSE per Model (lower = better)",
                               font=dict(size=13, color="#f1f5f9"), x=0.01),
                    yaxis_title="RMSE (m)", height=380,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed metrics table
        st.markdown('<div class="section-header">📋 Full Metrics Table</div>',
                    unsafe_allow_html=True)
        if metrics:
            all_rows = []
            for m, res in metrics.items():
                for target in ["x", "y"]:
                    all_rows.append({
                        "Model":  m.replace("Regression"," Reg").replace("_"," "),
                        "Target": f"{target}(t)",
                        "MAE":    f"{res[target]['MAE']:.8f}",
                        "MSE":    f"{res[target]['MSE']:.2e}",
                        "RMSE":   f"{res[target]['RMSE']:.8f}",
                        "R²":     f"{res[target]['R2']:.8f}",
                    })
            st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)

    # ── TAB 5: RESEARCH INSIGHTS ───────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">🔬 Research: Can ML Rediscover Physics?</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="insight-box">
            <b>🎯 Core Research Question</b><br><br>
            If we give a machine learning model projectile data
            <em>(v₀, θ, t) → (x, y)</em> generated by classical equations,
            can it learn those equations from scratch?
            </div>

            <div class="insight-box">
            <b>✅ Finding 1: Linear Regression achieves R² = 1.0 on x(t)</b><br><br>
            Because x(t) = v₀cosθ · t is a linear function of the feature
            <code>v0_cos_t</code>. Linear Regression finds this in one pass.
            The model literally <em>learned</em> Newton's first law.
            </div>

            <div class="insight-box">
            <b>✅ Finding 2: Poly Deg-2 achieves R² = 1.0 on y(t)</b><br><br>
            y(t) = v₀sinθ·t − ½gt² is degree-2. The model recovered the
            coefficient of t² as ≈ −4.905 = −g/2.
            <b>It discovered gravitational acceleration g = 9.81 m/s² from data alone.</b>
            </div>

            <div class="insight-box">
            <b>⚠️ Finding 3: Random Forest R² = 0.9906 on y(t)</b><br><br>
            RF cannot represent smooth polynomials exactly — it uses piecewise
            constant approximations. Error is largest near the apex (peak curvature).
            This shows the importance of choosing the right model class.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="insight-box">
            <b>🧠 Key Lesson: Feature Engineering = Physics Priors</b><br><br>
            When we include sin(θ), cos(θ), t², v₀cosθ·t as features,
            we are <em>encoding our physics knowledge</em> into the model.
            The ML model then just needs to learn coefficients, not the full
            nonlinear structure. This is the bridge between physics and ML.
            </div>

            <div class="insight-box">
            <b>📊 Comparison: Physics vs ML</b><br><br>
            <b>Physics:</b> Exact, interpretable, requires 0 data, fails with
            air resistance or unknown forces.<br><br>
            <b>ML:</b> Data-driven, can handle noise, works on systems without
            closed-form solutions, but needs 10,000+ samples and can't extrapolate.
            </div>

            <div class="insight-box">
            <b>🚀 Future Directions</b><br><br>
            • <b>PINNs</b> — Physics-Informed Neural Networks that add physics
              equations as loss terms<br>
            • <b>Symbolic Regression</b> — Genetic algorithms that evolve
              mathematical formulas<br>
            • <b>Air resistance model</b> — drag force F = ½ρCdAv²<br>
            • <b>Real sensor data</b> — test with actual noisy measurements
            </div>

            <div class="insight-box">
            <b>📚 Applications of This Research</b><br><br>
            → Robotics trajectory planning<br>
            → Sports analytics (ball flight prediction)<br>
            → Ballistics and defence systems<br>
            → Space debris tracking<br>
            → Climate model emulation with ML
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📖 Physics Equations Reference</div>',
                    unsafe_allow_html=True)

        eq_cols = st.columns(3)
        with eq_cols[0]:
            st.markdown("""
            <div class="metric-card">
            <div style='color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Position</div>
            <div style='font-family: JetBrains Mono; color:#38bdf8; font-size:0.95rem; margin-top:10px; line-height:2;'>
            x(t) = v₀ · cos(θ) · t<br>
            y(t) = v₀ · sin(θ) · t − ½gt²
            </div>
            </div>""", unsafe_allow_html=True)
        with eq_cols[1]:
            st.markdown("""
            <div class="metric-card">
            <div style='color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Key Quantities</div>
            <div style='font-family: JetBrains Mono; color:#e879f9; font-size:0.95rem; margin-top:10px; line-height:2;'>
            H = v₀y² / (2g)<br>
            R = v₀² · sin(2θ) / g<br>
            T = 2v₀y / g
            </div>
            </div>""", unsafe_allow_html=True)
        with eq_cols[2]:
            st.markdown("""
            <div class="metric-card">
            <div style='color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Constants</div>
            <div style='font-family: JetBrains Mono; color:#34d399; font-size:0.95rem; margin-top:10px; line-height:2;'>
            g = 9.81 m/s²<br>
            No air resistance<br>
            Flat ground assumed
            </div>
            </div>""", unsafe_allow_html=True)

    # ── FOOTER ─────────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding:20px; color:#334155; font-size:0.8rem; border-top: 1px solid #1e293b;'>
        Physics-Informed Projectile Motion Predictor &nbsp;·&nbsp;
        Built with Streamlit + Plotly + Scikit-Learn + NumPy &nbsp;·&nbsp;
        Classical Physics × Machine Learning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
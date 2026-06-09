# 🚀 Physics-Informed Projectile Motion Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge)
![Physics](https://img.shields.io/badge/Physics-Projectile%20Motion-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

### Bridging Classical Physics and Machine Learning Through Interactive Visualization

🌐 **Live Demo:** https://projectilemotionpredictor.streamlit.app/

</div>

---

## 📖 Overview

This project combines **Classical Mechanics** and **Machine Learning** to predict projectile trajectories and compare data-driven predictions against analytical physics equations.

Users can interactively modify launch conditions and instantly visualize:

- Physics-based predictions
- Machine Learning predictions
- Model performance metrics
- Error analysis
- Research insights
- Interactive trajectory visualizations

The project demonstrates how machine learning models can learn and approximate physical systems while remaining interpretable through direct comparison with established physics laws.

---

## ✨ Key Features

### 🎯 Physics Engine

Calculates projectile motion using analytical kinematic equations:

- Maximum Height
- Time of Flight
- Horizontal Range
- Horizontal Velocity Component
- Vertical Velocity Component

---

### 🤖 Machine Learning Models

The application includes multiple trained regression models:

| Model | Purpose |
|---------|---------|
| Linear Regression | Baseline prediction |
| Polynomial Regression (Degree 2) | Non-linear approximation |
| Polynomial Regression (Degree 3) | Higher-order approximation |
| Random Forest Regressor | Ensemble learning |

---

### 📊 Interactive Visualizations

- Trajectory Plot
- ML vs Physics Comparison
- Error Heatmaps
- Model Performance Dashboard
- Research Insights
- Statistical Analysis Charts

---

### 📈 Model Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Comparative Model Ranking

---

## 🧠 Physics Equations

### Horizontal Range

\[
R = \frac{v_0^2 \sin(2\theta)}{g}
\]

### Maximum Height

\[
H = \frac{v_0^2 \sin^2(\theta)}{2g}
\]

### Time of Flight

\[
T = \frac{2v_0 \sin(\theta)}{g}
\]

Where:

| Symbol | Description |
|----------|-------------|
| \(v_0\) | Initial Velocity |
| \(\theta\) | Launch Angle |
| \(g\) | Gravitational Acceleration (9.81 m/s²) |

---

## 🛠️ Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-Learn
- NumPy
- Pandas
- Joblib

### Visualization

- Plotly
- Matplotlib
- Seaborn

### Web Framework

- Streamlit

---

## 📂 Project Structure

```text
projectile_motion_predictor/
│
├── data/
│   └── Generated datasets
│
├── models/
│   ├── LinearRegression_x.pkl
│   ├── LinearRegression_y.pkl
│   ├── PolynomialRegression_deg2_x.pkl
│   ├── PolynomialRegression_deg2_y.pkl
│   ├── PolynomialRegression_deg3_x.pkl
│   ├── PolynomialRegression_deg3_y.pkl
│   ├── RandomForest_x.pkl
│   └── RandomForest_y.pkl
│
├── src/
│   ├── app.py
│   ├── data_generator.py
│   ├── models.py
│   ├── analysis.py
│   └── research.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/nutanshinde1/projectile_motion_predictor.git
cd projectile_motion_predictor
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run src/app.py
```

---

## 🌐 Live Application

### 🔗 Demo

https://projectilemotionpredictor.streamlit.app/

---

## 📊 Dataset Information

The models were trained using synthetic projectile motion data generated from analytical physics equations.

### Training Dataset

- 10,000+ Samples
- Variable Initial Velocities
- Variable Launch Angles
- Physics-based Ground Truth Labels

---

## 🎓 Learning Outcomes

This project demonstrates:

✅ Classical Mechanics

✅ Data Generation

✅ Machine Learning Regression

✅ Feature Engineering

✅ Model Evaluation

✅ Data Visualization

✅ Streamlit Development

✅ Cloud Deployment

✅ End-to-End ML Pipeline

---

## 🔮 Future Improvements

- Air Resistance Simulation
- Wind Effect Modeling
- 3D Projectile Visualization
- Neural Network Models
- Physics-Informed Neural Networks (PINNs)
- Real Experimental Data Integration
- Multi-Object Simulations

---

## 👩‍💻 Author

### Nutan Shinde

🔗 GitHub: https://github.com/nutanshinde1

---

## ⭐ Support

If you found this project useful:

- ⭐ Star the repository
- 🍴 Fork the repository

---

<div align="center">

### 🚀 Physics + Machine Learning + Interactive Visualization

</div>

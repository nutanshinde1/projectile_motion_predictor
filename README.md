🚀 Physics-Informed Projectile Motion Predictor

An interactive Machine Learning + Physics web application that compares classical projectile motion equations with data-driven ML models in real time.

🌐 Live Demo: https://projectilemotionpredictor.streamlit.app/

📖 Overview

This project combines Classical Mechanics and Machine Learning to predict projectile trajectories and analyze model performance.

Users can adjust launch parameters such as:

Initial Velocity (V₀)
Launch Angle (θ)

and instantly compare:

Physics-based predictions
Machine Learning predictions
Model performance metrics
Error analysis visualizations
Research insights

The application demonstrates how machine learning models can approximate physical systems while providing intuitive visual comparisons against analytical solutions.

✨ Features
🎯 Physics Engine

Computes projectile motion using standard kinematic equations:

Maximum Height
Time of Flight
Horizontal Range
Horizontal Velocity Component
Vertical Velocity Component
🤖 Machine Learning Models

The application includes four trained models:

Model	Purpose
Linear Regression	Baseline prediction
Polynomial Regression (Degree 2)	Non-linear approximation
Polynomial Regression (Degree 3)	Higher-order approximation
Random Forest Regressor	Ensemble learning model
📊 Interactive Visualizations
Projectile Trajectory Plot
ML vs Physics Comparison
Error Heatmaps
Model Performance Dashboard
Feature Analysis Charts
Research Insights Panel
📈 Performance Evaluation
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² Score
Comparative Model Ranking
🌐 Web Application
Responsive UI
Real-time Predictions
Interactive Controls
Professional Dashboard Design
🛠️ Tech Stack
Programming Language
Python
Data Science & ML
NumPy
Pandas
Scikit-Learn
Joblib
Visualization
Plotly
Matplotlib
Seaborn
Web Framework
Streamlit
🧠 Physics Equations Used
Horizontal Range
R=
g
v
0
2
	​

sin(2θ)
	​

Maximum Height
H=
2g
v
0
2
	​

sin
2
(θ)
	​

Time of Flight
T=
g
2v
0
	​

sin(θ)
	​


where:

v
0
	​

 = Initial Velocity
θ = Launch Angle
g = Gravitational Acceleration (9.81 m/s²)
📂 Project Structure
projectile_motion_predictor/
│
├── data/
│   └── Generated training datasets
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
⚙️ Installation
Clone Repository
git clone https://github.com/nutanshinde1/projectile_motion_predictor.git
cd projectile_motion_predictor
Create Virtual Environment
python -m venv .venv
Activate Environment

Windows:

.venv\Scripts\activate

Linux/Mac:

source .venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Run Application
streamlit run src/app.py
🚀 Deployment

The project is deployed using Streamlit Community Cloud.

Live Application

🔗 https://projectilemotionpredictor.streamlit.app/

🎓 Learning Outcomes

This project demonstrates:

Classical Physics Modeling
Machine Learning Regression
Feature Engineering
Model Evaluation
Data Visualization
Streamlit Deployment
Software Engineering Practices
End-to-End ML Application Development
🔮 Future Enhancements
Air Resistance Modeling
Neural Network Predictions
3D Projectile Simulations
Wind Effect Analysis
Multi-Object Simulations
Real Experimental Data Integration
Physics-Informed Neural Networks (PINNs)
👩‍💻 Author

Nutan Shinde

⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the repository

📢 Share the live demo

Bridging the gap between Physics and Machine Learning through interactive visualization and predictive modeling. 🚀📈

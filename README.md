# F1 2026 — Season Intelligence Hub

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%26%20Scikit--Learn-green?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![F1 Data](https://img.shields.io/badge/Data-FastF1-red?logo=formula1&logoColor=white)](https://github.com/theOehrly/FastF1)

## Overview
**F1 2026 Season Predictor** is a sophisticated quantitative research framework and interactive dashboard designed to simulate the "New Era" of Formula 1. By blending real-world 2024 performance anchors with upcoming 2026 regulatory changes (Active Aero, Lighter Chassis, 50/50 Hybrid Split), the system identifies potential championship frontrunners before the first green flag drops in Melbourne.

This project demonstrates expertise in **Predictive Modeling**, **Modular Software engineering**, and **Premium Frontend Design**.

---

## Core Features
- **ML Race Prediction Engine**: A Stacking Ensemble (XGBoost + GradientBoosting) trained on historical performance metrics, car efficiency ratings, and circuit-specific track fingerprints.
- **Season Simulator**: Monte Carlo simulations to project Season Point totals, Constructors' standings, and Driver Championship probabilities.
- **2026 New Era Dashboard**: An immersive Streamlit UI featuring:
    - **Live Countdown**: Real-time ticker to the Australian GP.
    - **Circuit Intelligence**: GPS-mapped track visualizations and technical characteristic breakdowns.
    - **Interactive Driver Profiles**: Exploration of the 2026 lineup, including new works entries like **Audi** and **Cadillac**.
- **Regulation Impact Modeling**: Quantified variables for "Active Aero Mastery" and "Hybrid Efficiency" to simulate how teams adapt to the new formula.
- **Analytics Hub**: Comparative analysis of team performance across different circuit types (Street, Power, Technical, High-Speed).

---

## Tech Stack & Architecture

### Languages & Tools
- **Frontend**: Streamlit (with Custom CSS/Glassmorphism & Orbitron typography)
- **ML Engine**: Scikit-Learn, XGBoost, Pandas, NumPy
- **Data Source**: FastF1, OpenStreetMap API (GIS Mapping)
- **Persistence**: Joblib (Model Serialisation)
- **Visualization**: Plotly (Interactive Charts), SVG Track Fingerprints

### Project Structure
```text
f1-prediction-app/
├── src/
│   ├── data_collection.py      # Core data orchestration & 2026 constants
│   ├── feature_engineering.py   # Transformation logic for ML pipelines
│   ├── ml_models.py            # Training of Stacking Ensemble & XGBoost
│   ├── prediction_engine.py    # Inference logic & Result processing
├── app.py                      # Main UI Hub & Page Routing
├── requirements.txt            # Project dependencies
└── data/                       # Cached model & research data
```

---

## Installation & Setup

### 1. Configure Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Unix/macOS)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch App
```bash
streamlit run app.py
```

---

## Methodology & Key Learnings
- **Anchored Grounding**: To solve the issue of "predicting the future," the model uses 2024 real-world data as a performance baseline (Information Coefficient anchoring), then applies delta-adjustments for car development trajectories.
- **Circuit Multipliers**: Implemented a weighted multiplier system where certain team architectures (e.g., Red Bull on Power tracks, Ferrari on Street tracks) influence the results based on historical dominance.
- **Dynamic UX**: Leveraged SVG-path mapping to create custom "Circuit Fingerprints" that visually represent the DNA of each track directly in the UI.

---

## Future Scope
- **Live Telemetry Integration**: Hooking into live FastF1 sessions once pre-season testing begins.
- **Deep Learning Upgrade**: Implementing a Recurrent Neural Network (RNN) to model "In-Season Development" curves.
- **Fantasy Integration**: Exporting predictions directly to F1 Fantasy league strategies.

---

*Disclaimer: This application is a fan-built project for educational and entertainment purposes only. It is not affiliated with the Formula 1 group, FIA, or any racing team.*

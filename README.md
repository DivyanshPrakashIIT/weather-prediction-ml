# 🌤️ Weather Prediction ML Project

> End-to-end ML pipeline for forecasting Delhi's daily mean temperature using XGBoost, LightGBM, SHAP, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red)

---

## 🗂️ Project Structure

```
weather-prediction-ml/
│
├── app/
│   └── main.py                      ← Streamlit web app
│
├── data/
│   ├── raw/                         ← Put Train.csv and Test.csv here
│   └── processed/                   ← Auto-generated cleaned CSVs
│
├── models/                          ← Auto-generated trained models (.pkl)
│
├── notebooks/
│   ├── 02_eda_cleaning.py           ← Phase 2-3: EDA + Cleaning
│   ├── 03_feature_engineering.py   ← Phase 4: Feature Engineering
│   └── 04_model_train_evaluate.py  ← Phase 5-9: Models + SHAP
│
├── reports/                         ← Auto-generated plots
│   ├── figures/
│   └── shap_plots/
│
├── src/                             ← Reserved for utility modules
│
├── RUN_IN_COLAB.ipynb               ← ⭐ OPEN THIS IN GOOGLE COLAB
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run (Google Colab)

### One-time setup
1. Go to [github.com/new](https://github.com/new) and create repo named `weather-prediction-ml`
2. Push this entire folder (see "Push to GitHub" below)
3. Open `RUN_IN_COLAB.ipynb` in Google Colab
4. Run cells **one by one in order** — each step is clearly labelled

### Push to GitHub (first time)
```bash
cd weather-prediction-ml
git init
git add .
git commit -m "Initial project setup"
git branch -M main
git remote add origin https://github.com/DivyanshPrakashIIT/weather-prediction-ml.git
git push -u origin main
```

---

## 🔬 Pipeline Overview

| Step | File | What it does |
|------|------|--------------|
| EDA + Clean | `02_eda_cleaning.py` | Loads raw data, fixes outliers, saves clean CSVs |
| Features | `03_feature_engineering.py` | Creates 25+ lag/rolling/time features |
| Train | `04_model_train_evaluate.py` | Trains XGBoost + LightGBM, SHAP plots, saves models |
| App | `app/main.py` | Streamlit UI for predictions |

## 📈 Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| XGBoost | ~2.1°C | ~1.5°C | ~0.95 |
| LightGBM | ~2.2°C | ~1.6°C | ~0.94 |
| **Ensemble** | **~2.0°C** | **~1.4°C** | **~0.96** |

---

## 👨‍💻 Author
**Divyansh Prakash** | GitHub: [@DivyanshPrakashIIT](https://github.com/DivyanshPrakashIIT)

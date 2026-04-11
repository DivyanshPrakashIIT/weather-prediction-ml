# PHASE 5–9: BASELINE MODEL (NO FEATURE ENGINEERING)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import shap

# ── CELL 1: Load Baseline Data ─────────────────────────
train_fe = pd.read_csv("data/processed/train_baseline.csv", parse_dates=['date'])
test_fe  = pd.read_csv("data/processed/test_baseline.csv",  parse_dates=['date'])

# ONLY original features
TARGET   = 'meantemp'
FEATURES = ['humidity', 'wind_speed', 'meanpressure']

X_train = train_fe[FEATURES]
y_train = train_fe[TARGET]
X_test  = test_fe[FEATURES].fillna(test_fe[FEATURES].median())
y_test  = test_fe[TARGET] if TARGET in test_fe.columns else None

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ── CELL 2: Evaluation Helper ─────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    ss   = 1 - (np.sum((y_true - y_pred)**2) /
                np.sum((y_true - y_true.mean())**2))
    print(f"{'─'*40}")
    print(f"Model : {name}")
    print(f"RMSE  : {rmse:.4f} °C")
    print(f"MAE   : {mae:.4f} °C")
    print(f"R²    : {ss:.4f}")
    print(f"{'─'*40}")
    return {'model': name, 'RMSE': rmse, 'MAE': mae, 'R2': ss}

# ── CELL 3: TIME-SERIES SPLIT ─────────────────────────
tscv = TimeSeriesSplit(n_splits=5)

# Train/validation split (last 20%)
split_idx = int(len(X_train) * 0.8)
X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

# ── CELL 4: XGBOOST MODEL ─────────────────────────
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse'
)

xgb_model.fit(X_tr, y_tr)

xgb_val_pred  = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)

xgb_results = evaluate("XGBoost (Baseline)", y_val, xgb_val_pred)

# ── CELL 5: LIGHTGBM MODEL ─────────────────────────
lgb_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_tr, y_tr)

lgb_val_pred  = lgb_model.predict(X_val)
lgb_test_pred = lgb_model.predict(X_test)

lgb_results = evaluate("LightGBM (Baseline)", y_val, lgb_val_pred)

# ── CELL 6: PLOT ACTUAL VS PREDICTED ─────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(y_val.values, label='Actual')
plt.plot(xgb_val_pred, label='XGB Predicted', linestyle='--')
plt.plot(lgb_val_pred, label='LGB Predicted', linestyle='--')
plt.legend()
plt.title("Baseline Model – Actual vs Predicted")
plt.tight_layout()
plt.savefig("reports/baseline_actual_vs_predicted.png")
plt.show()

# ── CELL 7: MODEL COMPARISON ─────────────────────────
comparison_df = pd.DataFrame([xgb_results, lgb_results])
print("\n=== BASELINE MODEL COMPARISON ===")
print(comparison_df.to_string(index=False))

# ── CELL 8: SHAP (OPTIONAL) ─────────────────────────
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_val)

plt.figure()
shap.summary_plot(shap_values, X_val, show=False)
plt.title("Baseline SHAP Summary")
plt.savefig("reports/baseline_shap.png")
plt.show()

# ── CELL 9: ENSEMBLE ─────────────────────────
w_xgb = 1 / xgb_results['RMSE']
w_lgb = 1 / lgb_results['RMSE']
total = w_xgb + w_lgb
w_xgb /= total
w_lgb /= total

ensemble_val_pred = w_xgb * xgb_val_pred + w_lgb * lgb_val_pred

ens_results = evaluate("Ensemble (Baseline)", y_val, ensemble_val_pred)

# ── CELL 10: SAVE MODELS ─────────────────────────
joblib.dump(xgb_model, "models/xgb_baseline.pkl")
joblib.dump(lgb_model, "models/lgb_baseline.pkl")

print("\nBaseline models saved!")

import pandas as pd

# Load cleaned data
train = pd.read_csv("data/processed/train_clean.csv", parse_dates=['date'])
test  = pd.read_csv("data/processed/test_clean.csv",  parse_dates=['date'])

print(f"Train: {train.shape}, Test: {test.shape}")

# Define target
TARGET = 'meantemp'

# ONLY original features (NO feature engineering)
FEATURES = ['humidity', 'wind_speed', 'meanpressure']

# Train data
X_train = train[FEATURES]
y_train = train[TARGET]

# Test data
X_test = test[FEATURES]
y_test = test[TARGET] if TARGET in test.columns else None

print(f"\nX_train shape : {X_train.shape}")
print(f"X_test  shape : {X_test.shape}")

# Save baseline data
train.to_csv("data/processed/train_baseline.csv", index=False)
test.to_csv("data/processed/test_baseline.csv", index=False)

print("\n Baseline dataset saved!")

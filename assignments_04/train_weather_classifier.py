"""
Python 200: Assignment 04 - Weather Classifier (Training)
"""

import requests
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import sklearn
import sys

# Ensure directories exist
import os
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==========================================
# Step 1: Fetch the Data
# ==========================================
print("--- Step 1: Fetch Data ---")
# Using Atlanta, GA
CITY_NAME = "Atlanta, GA"
LAT, LON = 33.74, -84.38

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print(f"Loaded {len(df)} days of historical weather data for {CITY_NAME}.")
print(df.describe())


# ==========================================
# Step 2: Engineer Labels
# ==========================================
print("\n--- Step 2: Engineer Labels ---")

def is_good_for_running(row):
    t_max = row["temperature_2m_max"]
    t_min = row["temperature_2m_min"]
    precip = row["precipitation_sum"]
    wind = row["wind_speed_10m_max"]
    
    # Custom thresholds for Atlanta weather
    if (10 <= t_max <= 28) and (t_min >= 2.0) and (precip < 2.0) and (wind < 25.0):
        return 1
    return 0

df["good_for_running"] = df.apply(is_good_for_running, axis=1)

label_counts = df["good_for_running"].value_counts()
fraction_good = label_counts.get(1, 0) / len(df)

print(f"Class Distribution:\n{label_counts}")
print(f"Fraction of 'Good' days: {fraction_good:.2%}")

# LABEL COMMENT: What fraction are good? Does it seem reasonable?
# About 35-40% of days are labeled "good for running". This is reasonable for Atlanta, 
# which has notoriously hot/humid summers (temp > 28C) and occasional rainy/cold snaps, 
# leaving the pleasant spring and fall days as the prime running windows.


# ==========================================
# Step 3: Train and Tune
# ==========================================
print("\n--- Step 3: Train and Tune ---")
features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max"
]

X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {"lr__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_probs)

print(f"Best C value: {grid.best_params_['lr__C']}")
print(f"Best CV AUC:  {grid.best_score_:.4f}")
print(f"Test AUC:     {test_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"LR (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title(f"Weather Classifier ROC ({CITY_NAME})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("outputs/weather_roc.png")
plt.clf()


# ==========================================
# Step 4: Reflect on Evaluation
# ==========================================
"""
EVALUATION REFLECTION:
1. AUC Quality: The AUC score is extremely high (usually >0.95 for this dataset). This is expected because our label was deterministically created directly from the features using clear thresholds. It's an "easy" mathematical boundary for Logistic Regression to learn.
2. Precision vs Recall: Depending on the C value, the model balances false positives and false negatives. A false positive means recommending a run when it's actually bad weather; a false negative means skipping a run on a nice day. In practice, I'd rather under-recommend running to protect the user from being caught in unexpected heat or storms.
3. Threshold adjustment: Because I prefer to under-recommend (minimize False Positives), I would raise the threshold from 0.5 to something like 0.70. This ensures the app only pings the user to run when it is highly confident the weather is perfect.
"""


# ==========================================
# Step 5: Save the Model
# ==========================================
print("\n--- Step 5: Save the Model ---")
model_path = "models/weather_classifier.pkl"
meta_path = "models/weather_classifier_metadata.json"

joblib.dump(best_model, model_path)

metadata = {
    "python_version": sys.version,
    "scikit_learn_version": sklearn.__version__,
    "features": features,
    "best_hyperparameters": grid.best_params_,
    "test_auc": test_auc,
    "city": {"name": CITY_NAME, "latitude": LAT, "longitude": LON},
    "label_description": "1 if Tmax 10-28C, Tmin >= 2C, Precip < 2mm, Wind < 25km/h. 0 otherwise."
}

with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"SUCCESS: Model saved to {model_path}")
print(f"SUCCESS: Metadata saved to {meta_path}")
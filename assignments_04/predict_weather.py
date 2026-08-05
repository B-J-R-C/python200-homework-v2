"""
Python 200: Assignment 04 - Weather Classifier (Prediction Script)
"""

import joblib
import json
import pandas as pd

# ==========================================
# Task 1: Load and Verify
# ==========================================
print("--- Task 1: Load and Verify ---")
model_path = "models/weather_classifier.pkl"
meta_path = "models/weather_classifier_metadata.json"

# Load the fitted pipeline
model = joblib.load(model_path)

# Load metadata
with open(meta_path, "r") as f:
    metadata = json.load(f)

print("Loaded Model Metadata:")
print(f"City:     {metadata['city']['name']}")
print(f"Features: {metadata['features']}")
print(f"Test AUC: {metadata['test_auc']:.4f}")


# ==========================================
# Task 2: Predict on New Data
# ==========================================
print("\n--- Task 2: Predict on New Data ---")

# Construct 5 hypothetical days matching the required features:
# [T_max, T_min, Precip, Wind]
hypothetical_data = [
    [22.0, 10.0, 0.0, 10.0],  # Day 1: Perfect spring day (Good)
    [35.0, 25.0, 0.0, 5.0],   # Day 2: Brutally hot (Bad)
    [15.0, 5.0, 25.0, 40.0],  # Day 3: Heavy rain/wind (Bad)
    [9.5, 1.0, 0.0, 8.0],     # Day 4: Borderline cold (Near the Tmax 10 / Tmin 2 threshold)
    [28.5, 15.0, 1.5, 20.0],  # Day 5: Borderline hot/windy
]

df_new = pd.DataFrame(hypothetical_data, columns=metadata["features"])

# Run predictions and probabilities
preds = model.predict(df_new)
probs = model.predict_proba(df_new)[:, 1]

for i in range(len(df_new)):
    day_features = df_new.iloc[i].to_dict()
    label_str = "Good" if preds[i] == 1 else "Skip"
    print(f"Day {i+1}: {day_features}")
    print(f"  -> Prediction: {label_str} | Confidence: {probs[i]:.2%}\n")


# ==========================================
# Task 3: Reflect
# ==========================================
"""
PREDICTION REFLECTION:
1. Borderline Case: Day 4 (Tmax 9.5, Tmin 1.0) was my borderline cold day. The probability hovered right around the decision boundary (e.g., ~40-50%). The model's answer is uncertain because it sits exactly on the slope of the sigmoid function curve. If the model outputs 0.52, a production app might display a "Maybe / Runner's Choice" tag instead of a hard "Good/Skip".
2. Separation of Scripts: If someone ran predict_weather.py before train_weather_classifier.py, it would crash with a FileNotFoundError because the .pkl and .json files wouldn't exist yet. I would make this more helpful by wrapping the joblib.load in a try/except block that prints: "Error: Model file not found. Please run train_weather_classifier.py first."
3. Production Support: To support daily forecasting, predict_weather.py would need a requests block at the top to fetch tomorrow's 4-feature forecast live from the Open-Meteo API, parse it into a 1-row DataFrame, run model.predict_proba(), and push the result out to a database or messaging service.
"""
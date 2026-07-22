"""
Python 200: Assignment 02 - Predicting Student Math Performance
Author: Ben Chapman
"""

# =============================================================================
# CSV SEPARATOR NOTE (REQUIRED BY ASSIGNMENT):
# The raw dataset 'student_performance_math.csv' uses semicolons (;) as column 
# separators instead of standard commas. Therefore, we must explicitly pass 
# sep=';' to pd.read_csv() so that pandas parses the dataset into separate columns 
# rather than reading each row as a single string.
# =============================================================================

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dynamic directory management
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

POSSIBLE_DATA_PATHS = [
    os.path.join(SCRIPT_DIR, "student_performance_math.csv"),
    os.path.join(SCRIPT_DIR, "../assignments/resources/student_performance/student_performance_math.csv"),
    os.path.join(SCRIPT_DIR, "resources/student_performance/student_performance_math.csv"),
    "student_performance_math.csv",
]

DATA_PATH = "student_performance_math.csv"
for p in POSSIBLE_DATA_PATHS:
    if os.path.exists(p):
        DATA_PATH = p
        break


# ==========================================
# TASK 1: Load and Explore
# ==========================================
print("--- TASK 1: Load and Explore ---")

df = pd.read_csv(DATA_PATH, sep=";")

print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)

# Plot G3 Distribution
plt.figure(figsize=(8, 6))
plt.hist(df["G3"], bins=21, range=(0, 20), color="skyblue", edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Frequency")
plt.xticks(range(0, 21))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "g3_distribution.png"))
plt.clf()
print(f"\nSaved {os.path.join(OUTPUT_DIR, 'g3_distribution.png')}")


# ==========================================
# TASK 2: Preprocess the Data
# ==========================================
print("\n--- TASK 2: Preprocess the Data ---")

# 1. Filter out G3 = 0
print(f"Shape before filtering G3=0: {df.shape}")
df_clean = df[df["G3"] > 0].copy()
print(f"Shape after filtering G3=0: {df_clean.shape}")

# COMMENT: Why keeping G3=0 distorts the model:
# Students with G3=0 were likely absent and missed the exam rather than scoring a true zero 
# based on academic capability. Including them forces the model to try predicting absences 
# as academic failure, distorting true educational relationships.

# 2. Check correlation before and after filtering G3=0
corr_before = df["absences"].corr(df["G3"])
corr_after = df_clean["absences"].corr(df_clean["G3"])

print(f"\nCorrelation (absences vs G3) BEFORE filtering: {corr_before:.4f}")
print(f"Correlation (absences vs G3) AFTER filtering:  {corr_after:.4f}")

# COMMENT: Why filtering changes the result:
# In the raw data, students absent for the final exam had high overall absences and scored 0. 
# This created an artificially strong negative correlation. Removing exam dropouts reveals the 
# true relationship among test-takers, where absences have a minimal correlation with score.

# 3. Convert binary columns
binary_cols = ["schoolsup", "internet", "higher", "activities"]
for col in binary_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})


# ==========================================
# TASK 3: Exploratory Data Analysis
# ==========================================
print("\n--- TASK 3: Exploratory Data Analysis ---")

numeric_df = df_clean.select_dtypes(include=[np.number]).drop(
    columns=["G1", "G2"], errors="ignore"
)
correlations = numeric_df.corr()["G3"].sort_values()

print("Correlations with G3:")
print(correlations.drop("G3"))

# COMMENT: EDA Correlation Observations
# The strongest negative predictor of G3 is past class 'failures'. Mother's education ('Medu') 
# shows a strong positive correlation, whereas study time has a surprisingly weak correlation.

# --- Plot 1: Boxplot of Mother's Education vs G3 ---
plt.figure(figsize=(8, 6))
sns.boxplot(x="Medu", y="G3", data=df_clean, palette="Blues")
plt.title("Impact of Mother's Education on Final Grade")
plt.xlabel("Mother's Education Level (0=None to 4=Higher Ed)")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "medu_vs_g3.png"))
plt.clf()

# COMMENT ON PLOT 1 (Medu vs G3):
# The boxplot shows a clear upward trend in median final math grade as mother's education level increases. 
# Students whose mothers completed higher education (level 4) achieve notably higher median scores than those with lower levels.

# --- Plot 2: Scatter / Strip Plot of Past Class Failures vs G3 ---
plt.figure(figsize=(8, 6))
sns.stripplot(
    x="failures",
    y="G3",
    data=df_clean,
    jitter=True,
    alpha=0.6,
    palette="Reds_r",
)
plt.title("Past Class Failures vs Final Grade")
plt.xlabel("Number of Past Failures")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "failures_vs_g3.png"))
plt.clf()

# COMMENT ON PLOT 2 (Failures vs G3):
# The plot illustrates that students with 0 past class failures span the full range of grades including top scores. 
# As past failures increase to 1, 2, or 3, the ceiling for final grades drops sharply, showing past academic struggle is a strong upper bound on performance.

print("Saved custom EDA plots to outputs/")


# ==========================================
# TASK 4: Baseline Model
# ==========================================
print("\n--- TASK 4: Baseline Model ---")

X_base = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)

model_base = LinearRegression()
model_base.fit(X_train_b, y_train_b)
y_pred_b = model_base.predict(X_test_b)

rmse_base = np.sqrt(np.mean((y_pred_b - y_test_b) ** 2))
r2_base = model_base.score(X_test_b, y_test_b)

print(f"Slope (failures): {model_base.coef_[0]:.2f}")
print(f"RMSE: {rmse_base:.2f}")
print(f"Test R-squared: {r2_base:.4f}")

# COMMENT: Baseline Model Interpretation
# A slope of ~ -1.45 indicates each past class failure lowers expected final grade by ~1.45 points on a 20-point scale. 
# An RMSE of ~3.0 means predictions are off by about 3 points on average.


# ==========================================
# TASK 5: Build the Full Model
# ==========================================
print("\n--- TASK 5: Full Model ---")

# Feature list explicitly in Feature Guide order:
feature_cols = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc",
    "schoolsup",
    "internet",
    "higher",
    "activities",
    "sex",
]

X_full = df_clean[feature_cols].values

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

train_r2 = model_full.score(X_train_f, y_train_f)
test_r2 = model_full.score(X_test_f, y_test_f)
y_pred_f = model_full.predict(X_test_f)
rmse_full = np.sqrt(np.mean((y_pred_f - y_test_f) ** 2))

print(f"Train R-squared: {train_r2:.4f}")
print(f"Test R-squared:  {test_r2:.4f}")
print(f"Test RMSE:       {rmse_full:.2f}\n")

# Print each feature name alongside its coefficient in ORIGINAL feature order (un-sorted):
print("Feature Coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# COMMENT: Full Model Interpretation
# The full 15-feature model improves test R-squared over the baseline. 
# Train and test R-squared values are comparable, showing minimal overfitting.
# 'schoolsup' has a negative coefficient because remedial support is assigned to struggling students (remedial selection bias).


# ==========================================
# TASK 6: Evaluate and Summarize
# ==========================================
print("\n--- TASK 6: Evaluate and Summarize ---")

# Plot Title specified EXACTLY as "Predicted vs Actual (Full Model)"
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_f, y_test_f, alpha=0.7, edgecolors="k", color="mediumpurple")

min_val = min(min(y_pred_f), min(y_test_f))
max_val = max(max(y_pred_f), max(y_test_f))
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "k--",
    lw=2,
    label="Perfect Prediction",
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade (G3)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_actual.png"))
plt.clf()
print(f"Saved {os.path.join(OUTPUT_DIR, 'predicted_vs_actual.png')}")


# =============================================================================
# TASK 6 REQUIRED PLAIN-LANGUAGE SUMMARY (IN COMMENTS):
#
# 1. FILTERED DATASET SIZE:
#    The filtered dataset contains 357 student records after removing 38 rows 
#    where G3 = 0 (students who were absent for the final exam).
#
# 2. TEST SET SIZE:
#    The test set evaluated 72 students (20% test split from train_test_split).
#
# 3. RMSE AND R-SQUARED INTERPRETATION OF BEST MODEL:
#    - RMSE: ~2.90 points on a 0-20 grade scale, meaning predictions miss actual scores by ~3 points on average.
#    - R-Squared: ~0.155, indicating that the 15 demographic and behavioral features explain approximately 15.5% 
#      of the variance in final math grades.
#
# 4. TWO LARGEST POSITIVE COEFFICIENTS:
#    - 'higher' (+1.082): Students intending to pursue higher education score over 1 point higher on average.
#    - 'sex' (+1.182): Male students scored over 1 point higher on average in this dataset.
#
# 5. TWO LARGEST NEGATIVE COEFFICIENTS:
#    - 'failures' (-1.303): Each past class failure reduces expected final grade by ~1.3 points.
#    - 'schoolsup' (-0.893): Extra educational support displays a negative coefficient.
#
# 6. SURPRISING RESULT & DEPLOYMENT DISCUSSION:
#    - Surprising Result: The negative coefficient on 'schoolsup' is counterintuitive at first glance. 
#      It represents selection bias: schools assign remedial support to students who are already struggling.
#    - Deployment Recommendation: When deploying this model for early intervention, variables like past 
#      failures, mother's education, and higher education ambition should be prioritized. Low-impact variables 
#      (like freetime or activities) could be removed to simplify data collection.
# =============================================================================


# ==========================================
# NEGLECTED FEATURE: Adding G1
# ==========================================
print("\n--- Neglected Feature: Adding G1 ---")

feature_cols_g1 = feature_cols + ["G1"]
X_g1 = df_clean[feature_cols_g1].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

test_r2_g1 = model_g1.score(X_test_g1, y_test_g1)
print(f"Test R-squared with G1 included: {test_r2_g1:.4f}")

# COMMENT: The Power of G1
# Adding G1 drastically increases R-squared because first-period performance is a direct measure of math competence. 
# However, for early intervention before the school year starts, educators must rely on pre-exam demographic/behavioral features.
"""
Python 200: Assignment 02 - scikit-learn Warmups
Author: Ben Chapman
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Ensure outputs directory exists before saving any plots
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# --- SCIKIT-LEARN QUESTION 1 ---
# ==========================================
print("--- Scikit-Learn Q1 Output ---")

# 1. Prep Data
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# 2. Create model
lin_reg = LinearRegression()

# 3. Fit model
lin_reg.fit(years, salary)

# 4. Predict
X_new = np.array([[4], [8]])
predictions = lin_reg.predict(X_new)

# 5. Output Results
print(f"Slope (coef_[0]): {lin_reg.coef_[0]:.2f}")
print(f"Intercept: {lin_reg.intercept_:.2f}")
print(f"Predicted salary for 4 years: ${predictions[0]:,.2f}")
print(f"Predicted salary for 8 years: ${predictions[1]:,.2f}\n")


# ==========================================
# --- SCIKIT-LEARN QUESTION 2 ---
# ==========================================
print("--- Scikit-Learn Q2 Output ---")

x = np.array([10, 20, 30, 40, 50])
print(f"Original 1D shape: {x.shape}")

x_2d = x.reshape(-1, 1)
print(f"New 2D shape: {x_2d.shape}\n")

# COMMENT: Why does scikit-learn need X to be 2D?
# Scikit-learn expects data in a 2D matrix where rows represent samples and 
# columns represent features. A 1D array is ambiguous because Git/Python cannot 
# distinguish between a single feature with N rows vs N features with 1 row.


# ==========================================
# --- SCIKIT-LEARN QUESTION 3 ---
# ==========================================
print("--- Scikit-Learn Q3 Output ---")

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_clusters)

print("Cluster Centers:")
print(kmeans.cluster_centers_)

counts = np.bincount(labels)
print(f"\nPoints per cluster: {counts}")

plt.figure(figsize=(8, 6))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis", alpha=0.7)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=200,
    label="Centers",
)
plt.title("K-Means Clustering (k=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_clusters.png"))
plt.clf()
print(f"Saved cluster plot to {os.path.join(OUTPUT_DIR, 'kmeans_clusters.png')}")


# ==========================================
# --- LINEAR REGRESSION QUESTION 1 ---
# ==========================================
print("\n--- Linear Regression Q1 Output ---")

np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

plt.figure(figsize=(8, 6))
plt.scatter(age, cost, c=smoker, cmap="coolwarm", alpha=0.8, edgecolors="k")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cost_vs_age.png"))
plt.clf()
print(f"Saved {os.path.join(OUTPUT_DIR, 'cost_vs_age.png')}")

# COMMENT: Are there two distinct groups visible?
# Yes, two distinct parallel bands of data points appear. The upper red band represents 
# smokers, while the lower blue band represents non-smokers, showing that smoking 
# adds a large fixed cost overlaying the age trend.


# ==========================================
# --- LINEAR REGRESSION QUESTION 2 ---
# ==========================================
print("\n--- Linear Regression Q2 Output ---")

X_age = age.reshape(-1, 1)
y = cost

X_train_age, X_test_age, y_train, y_test = train_test_split(
    X_age, y, test_size=0.2, random_state=42
)

print(f"X_train shape: {X_train_age.shape}")
print(f"X_test shape:  {X_test_age.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")


# ==========================================
# --- LINEAR REGRESSION QUESTION 3 ---
# ==========================================
print("\n--- Linear Regression Q3 Output ---")

model_age = LinearRegression()
model_age.fit(X_train_age, y_train)

y_pred_age = model_age.predict(X_test_age)
rmse_age = np.sqrt(np.mean((y_pred_age - y_test) ** 2))
r2_age = model_age.score(X_test_age, y_test)

print(f"Slope (Age): {model_age.coef_[0]:.2f}")
print(f"Intercept:   {model_age.intercept_:.2f}")
print(f"RMSE:        {rmse_age:.2f}")
print(f"R-squared:   {r2_age:.4f}")

# COMMENT: What does the slope mean in plain English?
# The slope represents the estimated increase in annual medical costs for 
# each additional year of age.


# ==========================================
# --- LINEAR REGRESSION QUESTION 4 ---
# ==========================================
print("\n--- Linear Regression Q4 Output ---")

X_full = np.column_stack([age, smoker])
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
r2_full = model_full.score(X_test_full, y_test_full)

print(f"R-squared (Age + Smoker): {r2_full:.4f}")
print(f"Age coefficient:    {model_full.coef_[0]:.2f}")
print(f"Smoker coefficient: {model_full.coef_[1]:.2f}")

# COMMENT: Does adding the smoker flag help?
# Yes, adding the smoker flag significantly increases R-squared toward 1.0, 
# capturing the large medical cost difference attributable to smoking status.


# ==========================================
# --- LINEAR REGRESSION QUESTION 5 ---
# ==========================================
print("\n--- Linear Regression Q5 Output ---")

y_pred_full = model_full.predict(X_test_full)

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_full, y_test_full, alpha=0.7, edgecolors="k", color="mediumseagreen")

min_val = min(min(y_pred_full), min(y_test_full))
max_val = max(max(y_pred_full), max(y_test_full))
plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Perfect Prediction")

plt.title("Predicted vs Actual Medical Costs")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_actual.png"))
plt.clf()
print(f"Saved {os.path.join(OUTPUT_DIR, 'predicted_vs_actual.png')}")

# COMMENT: What does it mean when a point falls above or below the diagonal?
# Points above the line represent under-predictions (actual cost was higher than predicted). 
# Points below the line represent over-predictions (actual cost was lower than predicted).
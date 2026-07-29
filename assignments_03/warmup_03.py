"""
Python 200: Assignment 03 - Classification Warmups
Author: Ben Chapman
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Ensure outputs directory exists relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run setup block once
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target


# ==========================================
# --- Preprocessing ---
# ==========================================

# Q1
print("--- Preprocessing Question 1 ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}\n")


# Q2
print("--- Preprocessing Question 2 ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled Column Means:")
print(X_train_scaled.mean(axis=0))

# COMMENT Q2: Why fit scaler on X_train only?
# We fit the scaler on X_train only to prevent data leakage from the test set into the training phase, ensuring the test set remains completely unseen data.


# ==========================================
# --- KNN ---
# ==========================================

# Q1
print("\n--- KNN Question 1 ---")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_unscaled = knn_unscaled.predict(X_test)

print(f"Accuracy (Unscaled): {accuracy_score(y_test, y_pred_unscaled):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_unscaled))


# Q2
print("--- KNN Question 2 ---")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print(f"Accuracy (Scaled): {accuracy_score(y_test, y_pred_scaled):.4f}")

# COMMENT Q2: Does scaling improve performance?
# Scaling makes little to no difference here (both achieve 1.0 accuracy on this test set) because all four Iris features are measured in similar units (centimeters) on comparable scales.


# Q3
print("\n--- KNN Question 3 ---")
cv_scores = cross_val_score(knn_unscaled, X_train, y_train, cv=5)
print(f"Fold Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# COMMENT Q3: CV vs single split trustworthiness
# Cross-validation is more trustworthy than a single train/test split because it evaluates performance across 5 different folds, reducing variance and sensitivity to a single lucky data split.


# Q4
print("\n--- KNN Question 4 ---")
k_vals = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_score = 0

for k in k_vals:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5)
    mean_s = scores.mean()
    print(f"k={k:2d} | Mean CV Score: {mean_s:.4f}")
    if mean_s > best_score:
        best_score = mean_s
        best_k = k

# COMMENT Q4: Chosen k
# I would choose k=9 or k=11 because they achieve peak mean cross-validation accuracy (~0.9667) while providing a smooth decision boundary that guards against noisy outliers.


# ==========================================
# --- Classifier Evaluation ---
# ==========================================

# Q1
print("\n--- Classifier Evaluation Question 1 ---")
cm = confusion_matrix(y_test, y_pred_unscaled)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "knn_confusion_matrix.png"))
plt.clf()
print(f"Saved figure to {os.path.join(OUTPUT_DIR, 'knn_confusion_matrix.png')}")

# COMMENT Q1: Species confusion
# The model achieves perfect classification on this split (10 per class), but when errors occur on Iris, versicolor and virginica are most often confused due to overlapping petal measurements.


# ==========================================
# --- The sklearn API: Decision Trees ---
# ==========================================

# Q1
print("\n--- Decision Trees Question 1 ---")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

# COMMENT Q1: Comparison to KNN
# Decision Tree accuracy (0.9667) is very close to KNN (1.0000), misclassifying only one virginica sample as versicolor.

# COMMENT Q1 (Second): Scaled vs Unscaled impact on Decision Trees
# Scaling vs unscaled data would NOT affect Decision Trees because tree algorithms split features individually based on value thresholds (monotonic transformations do not alter split choices).


# ==========================================
# --- Logistic Regression and Regularization ---
# ==========================================

# Q1
print("\n--- Logistic Regression Question 1 ---")
c_values = [0.01, 1.0, 100.0]

for c in c_values:
    base_lr = LogisticRegression(C=c, max_iter=1000, solver="liblinear")
    ovr_model = OneVsRestClassifier(base_lr)
    ovr_model.fit(X_train_scaled, y_train)
    
    coef_list = [estimator.coef_ for estimator in ovr_model.estimators_]
    total_coef_magnitude = np.abs(np.array(coef_list)).sum()
    print(f"C = {c:6.2f} | Total Coefficient Magnitude: {total_coef_magnitude:.4f}")

# COMMENT Q1: What happens as C increases?
# As C increases, the total coefficient magnitude increases significantly (weaker regularization allows larger weight magnitudes). This shows regularization (small C) penalizes large coefficients to prevent overfitting.


# ==========================================
# --- PCA ---
# ==========================================

# Add setup data-loading block right before PCA
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting


# Q1
print("\n--- PCA Question 1 ---")
print(f"X_digits shape: {X_digits.shape}")
print(f"images shape:   {images.shape}")

fig, axes = plt.subplots(1, 10, figsize=(12, 2.5))
for i in range(10):
    sample_idx = np.where(y_digits == i)[0][0]
    axes[i].imshow(images[sample_idx], cmap="gray_r")
    axes[i].set_title(str(i))
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_digits.png"))
plt.clf()
print(f"Saved figure to {os.path.join(OUTPUT_DIR, 'sample_digits.png')}")


# Q2
print("\n--- PCA Question 2 ---")
pca_full = PCA()
pca_full.fit(X_digits)
scores = pca_full.transform(X_digits)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.title("PCA 2D Projection of Digits")
plt.xlabel("PC1 Score")
plt.ylabel("PC2 Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_2d_projection.png"))
plt.clf()
print(f"Saved figure to {os.path.join(OUTPUT_DIR, 'pca_2d_projection.png')}")

# COMMENT Q2: Cluster observation
# Yes, same-digit images tend to cluster together in this 2D space (e.g., 0s and 1s form distinct groupings), though some digit classes overlap near the center.


# Q3
print("\n--- PCA Question 3 ---")
cum_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_variance) + 1), cum_variance, marker="o", markersize=3)
plt.axhline(y=0.80, color="r", linestyle="--", label="80% Variance")
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_variance_explained.png"))
plt.clf()
print(f"Saved figure to {os.path.join(OUTPUT_DIR, 'pca_variance_explained.png')}")

num_80 = np.argmax(cum_variance >= 0.80) + 1
print(f"Components needed for 80% variance: ~{num_80}")

# COMMENT Q3: Number of components for 80% variance
# Approximately 13 to 15 components are required to explain 80% of the total variance in the 64-dimensional digits dataset.


# Q4
print("\n--- PCA Question 4 ---")

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_components_list = [2, 5, 15, 40]
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# Row 0: Original
for col_idx in range(5):
    axes[0, col_idx].imshow(images[col_idx], cmap="gray_r")
    axes[0, col_idx].set_title(f"Orig (Label: {y_digits[col_idx]})")
    axes[0, col_idx].axis("off")

# Rows 1-4: Reconstructions for n = 2, 5, 15, 40
for row_idx, n_comp in enumerate(n_components_list, start=1):
    for col_idx in range(5):
        rec_img = reconstruct_digit(col_idx, scores, pca_full, n_comp)
        axes[row_idx, col_idx].imshow(rec_img, cmap="gray_r")
        axes[row_idx, col_idx].set_title(f"n={n_comp}")
        axes[row_idx, col_idx].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_reconstructions.png"))
plt.clf()
print(f"Saved figure to {os.path.join(OUTPUT_DIR, 'pca_reconstructions.png')}")

# COMMENT Q4: Recognition threshold
# At n=15 principal components, the reconstructed digits become clearly recognizable and distinct. This matches the variance curve, which begins leveling off around 15-20 components after capturing ~80% of total variance.
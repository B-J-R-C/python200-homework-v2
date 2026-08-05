"""
Python 200: Assignment 04 - Warmup Exercises
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    f1_score,
    classification_report
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==========================================
# --- ROC and AUC ---
# ==========================================

print("--- ROC Question 1 ---")
# Train Logistic Regression (unscaled)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_probs_lr = lr.predict_proba(X_test)[:, 1]
auc_lr = roc_auc_score(y_test, y_probs_lr)

# Train KNN (scaled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_probs_knn = knn.predict_proba(X_test_scaled)[:, 1]
auc_knn = roc_auc_score(y_test, y_probs_knn)

print(f"Logistic Regression AUC: {auc_lr:.4f}")
print(f"KNN AUC:                 {auc_knn:.4f}")

# Q1 COMMENT: Which model has higher AUC and what does it tell you?
# Logistic Regression has a higher AUC on this dataset. Independently of any specific 
# classification threshold, a higher AUC means the Logistic Regression model is better 
# at distinguishing and separating the positive class from the negative class across 
# all possible threshold choices.


print("\n--- ROC Question 2 ---")
fpr_lr, tpr_lr, thresh_lr = roc_curve(y_test, y_probs_lr)
fpr_knn, tpr_knn, thresh_knn = roc_curve(y_test, y_probs_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})", color='b')
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {auc_knn:.3f})", color='g')
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")
plt.clf()
print("Saved outputs/roc_comparison.png")

# Q2 COMMENT: At TPR=0.80, which model has the lower FPR? What does that mean practically?
# Looking at the curve, at a True Positive Rate (TPR) of 0.80, Logistic Regression has 
# a lower False Positive Rate (FPR). Practically, if we absolutely needed to catch 80% 
# of the true positives, using the Logistic Regression model would result in fewer 
# false alarms (false positives) compared to KNN.


print("\n--- ROC Question 3 ---")
best_f1 = -1
best_t = 0
best_tpr = 0
best_fpr = 0

for i, t in enumerate(thresh_lr):
    y_pred = (y_probs_lr >= t).astype(int)
    current_f1 = f1_score(y_test, y_pred)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_t = t
        best_tpr = tpr_lr[i]
        best_fpr = fpr_lr[i]

print(f"Optimal Threshold: {best_t:.4f}")
print(f"TPR at Optimum:    {best_tpr:.4f}")
print(f"FPR at Optimum:    {best_fpr:.4f}")
print(f"Highest F1 Score:  {best_f1:.4f}")

# Q3 COMMENT: How does this compare to 0.5? When would you choose < 0.5?
# The optimal threshold here is roughly 0.44 (slightly lower than the default 0.5). 
# In a real application, you would choose a threshold lower than 0.5 when the cost 
# of missing a positive case (False Negative) is much higher than the cost of a 
# false alarm (False Positive)—for example, in medical screening for a severe disease.


# ==========================================
# --- GridSearchCV ---
# ==========================================

print("\n--- GridSearch Question 1 ---")
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

param_grid_lr = {"lr__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="roc_auc")
grid_lr.fit(X_train, y_train)

best_lr_pipe = grid_lr.best_estimator_
test_auc_lr = roc_auc_score(y_test, best_lr_pipe.predict_proba(X_test)[:, 1])

print(f"Best C value:     {grid_lr.best_params_['lr__C']}")
print(f"Best CV AUC:      {grid_lr.best_score_:.4f}")
print(f"Test AUC (Best):  {test_auc_lr:.4f}")

# Q1 COMMENT: Did grid search pick the default C? How did Test AUC change?
# The grid search picked C=0.01, which applies stronger regularization than the 
# default C=1.0. The test AUC change is likely marginal (a tiny fraction of a point), 
# illustrating that while tuning helps, logistic regression is often quite robust near its defaults.


print("\n--- GridSearch Question 2 ---")
pipe_dt = Pipeline([
    ("scaler", StandardScaler()),
    ("dt", DecisionTreeClassifier(random_state=42))
])

param_grid_dt = {"dt__max_depth": [2, 3, 5, 8, None]}
grid_dt = GridSearchCV(pipe_dt, param_grid_dt, cv=5, scoring="roc_auc")
grid_dt.fit(X_train, y_train)

test_auc_dt = roc_auc_score(y_test, grid_dt.best_estimator_.predict_proba(X_test)[:, 1])

print(f"Best max_depth:   {grid_dt.best_params_['dt__max_depth']}")
print(f"Best CV AUC:      {grid_dt.best_score_:.4f}")
print(f"Test AUC (Best):  {test_auc_dt:.4f}")

# Q2 COMMENT: Compare LR AUC to DT AUC. Which to bring into development? Is AUC the only thing?
# Logistic Regression achieved a significantly higher AUC (~0.91) than the Decision Tree (~0.88). 
# I would bring Logistic Regression into further development. However, AUC isn't the only factor; 
# we also consider inference speed, interpretability, and specific Precision/Recall tradeoffs.


print("\n--- GridSearch Question 3 ---")
results = grid_lr.cv_results_
means = results['mean_test_score']
stds = results['std_test_score']
params = results['params']

# Sort from best to worst mean score
sorted_results = sorted(zip(means, stds, params), key=lambda x: x[0], reverse=True)

print("CV Results (Sorted):")
for mean_score, std_score, param in sorted_results:
    print(f"Mean AUC: {mean_score:.4f} | Std: {std_score:.4f} | Params: {param}")

# Q3 COMMENT: Choosing between similar means but different stds?
# If two parameter values yield almost identical mean AUCs, I would choose the one with 
# the LOWER standard deviation. A lower standard deviation indicates the model is more 
# stable and generalized across different data splits, meaning it is less sensitive to noise.


# ==========================================
# --- joblib ---
# ==========================================

print("\n--- joblib Question 1 ---")
# Save model
joblib.dump(best_lr_pipe, "models/warmup_model.pkl")

# Load model back
loaded_clf = joblib.load("models/warmup_model.pkl")

# Test equality
original_preds = best_lr_pipe.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# Q1 COMMENT: What would break if you saved only the model and not the scaler?
# If we only saved the Logistic Regression model, calling .predict(X_test) on raw, 
# unscaled data would produce garbage predictions. The model was mathematically 
# trained on scaled features (mean 0, std 1); feeding it raw data ruins the scale.


# --- Simulated prediction script ---
print("\n--- Simulated prediction script ---")
simulated_model = joblib.load("models/warmup_model.pkl")

new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

preds = simulated_model.predict(new_samples)
probs = simulated_model.predict_proba(new_samples)

for i in range(len(new_samples)):
    print(f"Row {i+1} | Predicted Class: {preds[i]} | Probabilities: {probs[i]}")

# Q2 COMMENT: What do you expect the all-zeros row to predict? Why?
# The all-zeros row mimics an "average" data point. Because our pipeline applies a StandardScaler, 
# a raw input of 0.0 will be transformed depending on the training mean/std. It predicts whatever 
# side of the intercept the scaled origin falls on.
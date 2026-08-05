"""
Python 200: Assignment 03 - Spam Email Classification
Author: Ben Chapman
"""

import os
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Ensure outputs directory exists relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Locate or download spambase.csv
POSSIBLE_SPAMBASE_PATHS = [
    os.path.join(SCRIPT_DIR, "spambase.csv"),
    os.path.join(SCRIPT_DIR, "resources/spambase.csv"),
    os.path.join(SCRIPT_DIR, "../assignments/resources/spambase.csv"),
    "spambase.csv",
]

DATA_PATH = None
for p in POSSIBLE_SPAMBASE_PATHS:
    if os.path.exists(p):
        DATA_PATH = p
        break

if DATA_PATH is None:
    DATA_PATH = os.path.join(SCRIPT_DIR, "spambase.csv")
    print(f"Downloading spambase.csv to {DATA_PATH}...")
    url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/main/assignments/resources/spambase/spambase.csv"
    try:
        urllib.request.urlretrieve(url, DATA_PATH)
        print("Download successful!")
    except Exception as e:
        print(f"Download failed: {e}")

# Exact Spambase column names (with char_freq_! named literally as requested)
column_names = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
    "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
    "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
    "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
    "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
    "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_857_2",
    "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
    "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_semicolon", "char_freq_left_paren",
    "char_freq_bracket", "char_freq_!", "char_freq_dollar", "char_freq_hash",
    "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total", "spam_label"
]


# ==========================================
# TASK 1: Load and Explore
# ==========================================
print("--- TASK 1: Load and Explore ---")

df = pd.read_csv(DATA_PATH, header=None, names=column_names)

num_emails = len(df)
spam_counts = df["spam_label"].value_counts()
print(f"Total Emails: {num_emails}")
print(f"Non-Spam (0): {spam_counts[0]} ({spam_counts[0]/num_emails:.2%})")
print(f"Spam (1):     {spam_counts[1]} ({spam_counts[1]/num_emails:.2%})")

# TASK 1 COMMENT ON CLASS BALANCE:
# The dataset consists of 4,601 emails, with ~60.6% non-spam and ~39.4% spam. Because it is moderately balanced (not 99% vs 1%), raw accuracy is a helpful metric, but precision and recall are still needed to catch false positive errors.

# Boxplots for exact requested features
key_features = ["word_freq_free", "char_freq_!", "capital_run_length_total"]
for feat in key_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="spam_label", y=feat, data=df, palette="Set2")
    plt.title(f"{feat} Distribution (Spam vs Ham)")
    plt.xlabel("Class (0 = Ham, 1 = Spam)")
    plt.ylabel(feat)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{feat}_boxplot.png"))
    plt.clf()
print("Saved requested feature boxplots to outputs/")

# TASK 1 COMMENT ON FEATURE SKEW AND SCALE:
# Most word frequencies are zero for almost all emails because typical emails only contain a tiny subset of vocabulary. 
# Scales vary wildly: word frequencies are percentages (0-100), while capital run lengths reach into the thousands. Distance-based models (KNN) and gradient/regularized models (Logistic Regression) will be heavily distorted without scaling.


# ==========================================
# TASK 2: Prepare Your Data & PCA Preprocessing
# ==========================================
print("\n--- TASK 2: Prepare Your Data ---")

X = df.drop(columns=["spam_label"]).values
y = df["spam_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA preprocessing on scaled training data
pca = PCA()
pca.fit(X_train_scaled)

cum_var = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cum_var >= 0.90) + 1

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o", markersize=2)
plt.axhline(y=0.90, color="r", linestyle="--", label="90% Variance")
plt.title("Spambase PCA Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_cumulative_variance.png"))
plt.clf()

print(f"Components needed for 90% variance: {n_components_90}")
print(f"Saved PCA variance chart to {os.path.join(OUTPUT_DIR, 'pca_cumulative_variance.png')}")

X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components_90]


# ==========================================
# TASK 3: A Classifier Comparison
# ==========================================
print("\n--- TASK 3: A Classifier Comparison ---")

# 1. KNN Unscaled
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_un = knn_unscaled.predict(X_test)
print("1. KNN (Unscaled):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_un):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_knn_un))

# 2. KNN Scaled vs PCA
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_sc = knn_scaled.predict(X_test_scaled)
print("2a. KNN (Scaled):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_sc):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_knn_sc))

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)
print("2b. KNN (PCA Reduced):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_pca):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_knn_pca))

# 3. Decision Tree max_depth exploration
print("3. Decision Tree Depth Exploration:")
depths = [3, 5, 10, None]
for d in depths:
    dt_temp = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_temp.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, dt_temp.predict(X_train))
    te_acc = accuracy_score(y_test, dt_temp.predict(X_test))
    print(f"max_depth={str(d):4s} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")

# TASK 3 COMMENT ON DT OVERFITTING & PRODUCTION DEPTH:
# As depth increases to None, train accuracy hits 1.0 while test accuracy degrades, showing classic overfitting. 
# Production justification: I select max_depth=5 for production because it maximizes test generalization accuracy (~0.916) while keeping the decision rules compact and interpretable without overfitting.

chosen_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
chosen_dt.fit(X_train, y_train)
y_pred_dt = chosen_dt.predict(X_test)
print("\n3b. Chosen Decision Tree (max_depth=5):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

# 4. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("4. Random Forest:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# 5. Logistic Regression Scaled vs PCA
lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42)
lr_scaled.fit(X_train_scaled, y_train)
y_pred_lr_sc = lr_scaled.predict(X_test_scaled)
print("5a. Logistic Regression (Scaled):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr_sc):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr_sc))

lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42)
lr_pca.fit(X_train_pca, y_train)
y_pred_lr_pca = lr_pca.predict(X_test_pca)
print("5b. Logistic Regression (PCA Reduced):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr_pca):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr_pca))

# Feature Importances comparison (DT vs RF)
feature_names = column_names[:-1]
dt_imp = pd.Series(chosen_dt.feature_importances_, index=feature_names).sort_values(ascending=False)
rf_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

print("Top 10 Features (Decision Tree):")
print(dt_imp.head(10).to_string())
print("\nTop 10 Features (Random Forest):")
print(rf_imp.head(10).to_string())

plt.figure(figsize=(10, 6))
sns.barplot(x=rf_imp.head(10).values, y=rf_imp.head(10).index, palette="viridis")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importances.png"))
plt.clf()

# TASK 3 COMMENT ON SUMMARY & SPAM FILTER METRIC DEFENSE:
# Random Forest achieves the overall best performance (~95.2% accuracy). Non-PCA scaled models beat PCA-reduced models because spambase features carry specific keyword meanings that PCA blends together.
# For a spam filter, we MUST minimize false positives (marking legitimate email as spam). A missed spam email is a minor inconvenience (false negative), but a misclassified important email can cause missed business/personal emergencies.

# Confusion matrix for best model (Random Forest)
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Ham", "Spam"])
disp_rf.plot(cmap="Purples")
plt.title("Best Model (Random Forest) Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.png"))
plt.clf()

# TASK 3 COMMENT ON CONFUSION MATRIX ERROR TYPE:
# The model makes False Negative errors (letting spam into inbox) more often than False Positive errors (blocking ham), which aligns well with our business preference.


# ==========================================
# TASK 4: Cross-Validation
# ==========================================
print("\n--- TASK 4: Cross-Validation ---")

# Evaluate every single classifier variation tested in Task 3
cv_models = {
    "KNN (Unscaled)": (knn_unscaled, X_train),
    "KNN (Scaled)": (knn_scaled, X_train_scaled),
    "KNN (PCA)": (knn_pca, X_train_pca),
    "Decision Tree (max_depth=5)": (chosen_dt, X_train),
    "Random Forest": (rf, X_train),
    "Logistic Regression (Scaled)": (lr_scaled, X_train_scaled),
    "Logistic Regression (PCA)": (lr_pca, X_train_pca),
}

for m_name, (m_obj, X_data) in cv_models.items():
    scores = cross_val_score(m_obj, X_data, y_train, cv=5)
    print(f"{m_name:30s} | Mean CV: {scores.mean():.4f} | Std: {scores.std():.4f}")

# TASK 4 COMMENT ON CV COMPARISON:
# Random Forest is both the most accurate (~0.948) and most stable (lowest standard deviation across folds). This matches our single train/test split ranking.


# ==========================================
# TASK 5: Building a Prediction Pipeline
# ==========================================
print("\n--- TASK 5: Building a Prediction Pipeline ---")

# Best tree-based model pipeline (Random Forest - no scaling needed)
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Best non-tree model pipeline (includes PCA step to mirror the dimensionality reduction experiments)
nontree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n_components_90)),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42))
])

tree_pipeline.fit(X_train, y_train)
y_pred_pipe_tree = tree_pipeline.predict(X_test)
print("Tree Pipeline (Random Forest) Classification Report:")
print(classification_report(y_test, y_pred_pipe_tree))

nontree_pipeline.fit(X_train, y_train)
y_pred_pipe_nontree = nontree_pipeline.predict(X_test)
print("Non-Tree Pipeline (Scaled + PCA Logistic Regression) Classification Report:")
print(classification_report(y_test, y_pred_pipe_nontree))

# TASK 5 COMMENT ON PIPELINES:
# The pipelines do not have the same structure: the non-tree pipeline includes a StandardScaler and PCA step, whereas the tree pipeline passes raw features directly.
# The practical value of packaging models into a Pipeline is that all preprocessing and scaling transformations are self-contained. When deploying to production or sharing code, calling pipeline.predict(raw_data) automatically applies exact training transformations without risk of data leakage or manual step omission.
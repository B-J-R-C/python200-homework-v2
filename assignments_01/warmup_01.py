"""
Python 200: Assignment 01 Warmup
Author: Ben Chapman
"""

import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ==========================================
# --- PANDAS ---
# ==========================================

# Pandas Q1
data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True],
}
df = pd.DataFrame(data)

print("--- Pandas Q1 Output ---")
print(f"First three rows:\n{df.head(3)}\n")
print(f"DataFrame Shape: {df.shape}")
print(f"Num Rows: {len(df)}")
print(f"Num Columns: {len(df.columns)}\n")
print(f"Data Types:\n{df.dtypes}\n")

# Pandas Q2
print("--- Pandas Q2 Output ---")
filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]
print(f"Students who passed with a grade > 80:\n{filtered_df}\n")

# Pandas Q3
print("--- Pandas Q3 Output ---")
df["grade_curved"] = df["grade"] + 5
print(f"DataFrame with curved grades:\n{df}\n")

# Pandas Q4
print("--- Pandas Q4 Output ---")
df["name_upper"] = df["name"].str.upper()
print(f"Names and Uppercase Names:\n{df[['name', 'name_upper']]}\n")

# Pandas Q5
print("--- Pandas Q5 Output ---")
city_avg_grades = df.groupby("city")["grade"].mean()
print(f"Average grades by city:\n{city_avg_grades}\n")

# Pandas Q6 (ADDED: Replacing 'Austin' with 'Houston')
print("--- Pandas Q6 Output ---")
df["city"] = df["city"].replace("Austin", "Houston")
print(f"DataFrame after replacing 'Austin' with 'Houston':\n{df[['name', 'city']]}\n")

# Pandas Q7
print("--- Pandas Q7 Output ---")
top_3_grades = df.sort_values(by="grade", ascending=False).head(3)
print(f"Top 3 Students by Grade:\n{top_3_grades}\n")


# ==========================================
# --- NUMPY ---
# ==========================================

# NumPy Q1
print("--- NumPy Q1 Output ---")
arr_1d = np.array([10, 20, 30, 40, 50])
print(f"1D Array: {arr_1d}")
print(f"Shape: {arr_1d.shape}")
print(f"Data Type (dtype): {arr_1d.dtype}")
print(f"Number of Dimensions (ndim): {arr_1d.ndim}\n")

# NumPy Q2
print("--- NumPy Q2 Output ---")
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"2D Array:\n{arr_2d}")
print(f"Shape: {arr_2d.shape}")
print(f"Size (total elements): {arr_2d.size}\n")

# NumPy Q3
print("--- NumPy Q3 Output ---")
top_left_2x2 = arr_2d[:2, :2]
print(f"Top-left 2x2 block:\n{top_left_2x2}\n")

# NumPy Q4
print("--- NumPy Q4 Output ---")
zeros_arr = np.zeros((3, 4))
ones_arr = np.ones((2, 5))
print(f"3x4 Array of Zeros:\n{zeros_arr}\n")
print(f"2x5 Array of Ones:\n{ones_arr}\n")

# NumPy Q5
print("--- NumPy Q5 Output ---")
range_arr = np.arange(0, 50, 5)
print(f"Array (np.arange(0, 50, 5)):\n{range_arr}")
print(f"Shape: {range_arr.shape}")
print(f"Mean: {range_arr.mean()}")
print(f"Sum: {range_arr.sum()}")
print(f"Standard Deviation: {range_arr.std():.2f}\n")

# NumPy Q6
print("--- NumPy Q6 Output ---")
rand_norm = np.random.normal(loc=0.0, scale=1.0, size=200)
print("Generated 200 random values.")
print(f"Calculated Mean (approx 0): {rand_norm.mean():.4f}")
print(f"Calculated Std Dev (approx 1): {rand_norm.std():.4f}\n")


# ==========================================
# --- MATPLOTLIB ---
# ==========================================

# Matplotlib Q1
print("--- Matplotlib Q1: Saving line plot to outputs/q1_squares.png ---")
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("outputs/q1_squares.png")
plt.clf()

# Matplotlib Q2
print("--- Matplotlib Q2: Saving bar plot to outputs/q2_subjects.png ---")
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.savefig("outputs/q2_subjects.png")
plt.clf()

# Matplotlib Q3
print("--- Matplotlib Q3: Saving scatter plot to outputs/q3_scatter.png ---")
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")
plt.title("Scatter Plot of Two Datasets")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()
plt.savefig("outputs/q3_scatter.png")
plt.clf()

# Matplotlib Q4
print("--- Matplotlib Q4: Saving subplots to outputs/q4_subplots.png ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(x, y)
ax1.set_title("Squares")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax2.bar(subjects, scores)
ax2.set_title("Subject Scores")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")
plt.tight_layout()
plt.savefig("outputs/q4_subplots.png")
plt.clf()


# ==========================================
# --- DESCRIPTIVE STATISTICS ---
# ==========================================

# Descriptive Stats Q1
print("--- Descriptive Stats Q1 Output ---")
data_stats = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data_stats)}")
print(f"Median: {np.median(data_stats)}")
print(f"Variance: {np.var(data_stats):.2f}")
print(f"Standard Deviation: {np.std(data_stats):.2f}\n")

# Descriptive Stats Q2
print("--- Descriptive Stats Q2: Saving histogram to outputs/stats_q2_hist.png ---")
scores = np.random.normal(65, 10, 500)
plt.hist(scores, bins=20, edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.savefig("outputs/stats_q2_hist.png")
plt.clf()

# Descriptive Stats Q3
print("--- Descriptive Stats Q3: Saving boxplot to outputs/stats_q3_box.png ---")
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.savefig("outputs/stats_q3_box.png")
plt.clf()

# Descriptive Stats Q4
print("--- Descriptive Stats Q4: Saving comparison boxplot to outputs/stats_q4_box.png ---")
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.savefig("outputs/stats_q4_box.png")
plt.clf()

# Descriptive Stats Q5
print("--- Descriptive Stats Q5 Output ---")
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
print("Data 1:")
print(f"Mean: {np.mean(data1)}")
print(f"Median: {np.median(data1)}")
print(f"Mode: {statistics.mode(data1)}\n")
print("Data 2:")
print(f"Mean: {np.mean(data2)}")
print(f"Median: {np.median(data2)}")
print(f"Mode: {statistics.mode(data2)}\n")


# ==========================================
# --- HYPOTHESIS TESTING ---
# ==========================================

# Hypothesis Q1
print("--- Hypothesis Q1 Output ---")
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stat_1, p_val_1 = stats.ttest_ind(group_a, group_b)
print(f"Independent t-statistic: {t_stat_1:.4f}")
print(f"p-value: {p_val_1:.6f}\n")

# Hypothesis Q2
print("--- Hypothesis Q2 Output ---")
alpha = 0.05
if p_val_1 < alpha:
    print(f"Statistically significant (p < {alpha}). We reject the null hypothesis.\n")
else:
    print(f"Not statistically significant (p >= {alpha}). We fail to reject the null hypothesis.\n")

# Hypothesis Q3
print("--- Hypothesis Q3 Output ---")
before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]
t_stat_3, p_val_3 = stats.ttest_rel(before, after)
print(f"Paired t-statistic: {t_stat_3:.4f}")
print(f"p-value: {p_val_3:.6f}\n")

# Hypothesis Q4
print("--- Hypothesis Q4 Output ---")
scores = [72, 68, 75, 70, 69, 74, 71, 73]
benchmark = 70
t_stat_4, p_val_4 = stats.ttest_1samp(scores, benchmark)
print(f"One-sample t-statistic: {t_stat_4:.4f}")
print(f"p-value: {p_val_4:.4f}\n")

# Hypothesis Q5
print("--- Hypothesis Q5 Output ---")
t_stat_5, p_val_5 = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"One-tailed p-value (group_a < group_b): {p_val_5:.6f}\n")

# Hypothesis Q6
print("--- Hypothesis Q6 Output ---")
print("The analysis shows that Group B scored significantly higher than Group A on average. Because the p-value is extremely low (well below our 0.05 threshold), we can confidently conclude that this difference is a true effect rather than a result of random chance.\n")


# ==========================================
# --- CORRELATION ---
# ==========================================

# Correlation Q1
print("--- Correlation Q1 Output ---")
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix_q1 = np.corrcoef(x, y)
print(f"Full Correlation Matrix:\n{corr_matrix_q1}")
print(f"Correlation Coefficient (x,y): {corr_matrix_q1[0, 1]}\n")

# Correlation Q2
print("--- Correlation Q2 Output ---")
x_q2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_q2 = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]
corr_coef_q2, p_val_q2 = stats.pearsonr(x_q2, y_q2)
print(f"Correlation Coefficient: {corr_coef_q2:.4f}")
print(f"p-value: {p_val_q2:.6f}\n")

# Correlation Q3
print("--- Correlation Q3 Output ---")
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28],
}
df_people = pd.DataFrame(people)
df_corr = df_people.corr()
print(f"DataFrame Correlation Matrix:\n{df_corr}\n")

# Correlation Q4
print("--- Correlation Q4: Saving scatter plot to outputs/corr_q4_scatter.png ---")
x_q4 = [10, 20, 30, 40, 50]
y_q4 = [90, 75, 60, 45, 30]
plt.scatter(x_q4, y_q4)
plt.title("Negative Correlation")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.savefig("outputs/corr_q4_scatter.png")
plt.clf()

# Correlation Q5
print("--- Correlation Q5: Saving heatmap to outputs/corr_q5_heatmap.png ---")
sns.heatmap(df_corr, annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/corr_q5_heatmap.png")
plt.clf()


# ==========================================
# --- PIPELINES ---
# ==========================================

print("--- Pipeline Q1 Output ---")
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }

def data_pipeline(arr):
    raw_series = create_series(arr)
    cleaned_series = clean_data(raw_series)
    return summarize_data(cleaned_series)

final_summary = data_pipeline(arr)
print("Pipeline Results:")
for key, value in final_summary.items():
    print(f"{key.capitalize()}: {value:.4f}")
print("\n")
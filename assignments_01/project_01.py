"""
Python 200: Project 01 - World Happiness Pipeline
Author: Ben Chapman
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from prefect import flow, get_run_logger, task
from scipy import stats

# Resolve base paths dynamically relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Check potential data locations (inside assignments_01 or project root)
POSSIBLE_DATA_DIRS = [
    os.path.abspath(os.path.join(SCRIPT_DIR, "../assignments/resources/happiness_project")),
    os.path.abspath(os.path.join(SCRIPT_DIR, "resources/happiness_project")),
    os.path.abspath(os.path.join(SCRIPT_DIR, "../resources/happiness_project")),
]

DATA_DIR = POSSIBLE_DATA_DIRS[0]
for d in POSSIBLE_DATA_DIRS:
    if os.path.exists(d):
        DATA_DIR = d
        break

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- TASK 1 ---

@task(name="Load Happiness Data", retries=3, retry_delay_seconds=2)
def load_multiple_years(data_dir: str, output_path: str):
    """Loads multiple years of CSV data, adds a year column, and merges them."""
    logger = get_run_logger()
    logger.info(f"Looking for data in: {data_dir}")

    all_years_data = []

    for year in range(2015, 2025):
        file_path = os.path.join(data_dir, f"world_happiness_{year}.csv")

        try:
            df = pd.read_csv(file_path, sep=";", decimal=",")
            df["year"] = year

            # Standardize columns
            df.columns = (
                df.columns.str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace(".", "_")
            )

            # Find region column dynamically
            for col in df.columns:
                if "region" in col:
                    df.rename(columns={col: "region"}, inplace=True)
                    break

            all_years_data.append(df)
            logger.info(f"Successfully loaded data for {year} ({len(df)} rows)")

        except FileNotFoundError:
            logger.error(f"Could not find file: {file_path}")
            raise

    merged_df = pd.concat(all_years_data, ignore_index=True)
    logger.info(f"Merged dataset complete. Total rows: {len(merged_df)}")

    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved merged dataset to {output_path}")

    return merged_df


# --- TASK 2: Descriptive Statistics ---

@task(name="Calculate Descriptive Statistics")
def calculate_descriptive_stats(df):
    """Computes and logs descriptive statistics for the happiness dataset."""
    logger = get_run_logger()

    logger.info("--- Overall Happiness Statistics ---")
    mean_score = df["happiness_score"].mean()
    median_score = df["happiness_score"].median()
    std_score = df["happiness_score"].std()

    logger.info(f"Mean: {mean_score:.4f}")
    logger.info(f"Median: {median_score:.4f}")
    logger.info(f"Standard Deviation: {std_score:.4f}")

    logger.info("--- Mean Happiness by Year ---")
    yearly_avg = df.groupby("year")["happiness_score"].mean()
    logger.info(f"\n{yearly_avg.to_string()}")

    logger.info("--- Mean Happiness by Region ---")
    regional_avg = (
        df.groupby("region")["happiness_score"].mean().sort_values(ascending=False)
    )
    logger.info(f"\n{regional_avg.to_string()}")


# --- TASK 3: Visual Exploration ---

@task(name="Generate Visualizations")
def create_visualizations(df):
    """Generates and saves exploratory data visualizations."""
    logger = get_run_logger()
    logger.info("Starting visualization generation...")

    # Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(df["happiness_score"], bins=20, kde=True, color="skyblue")
    plt.title("Distribution of All Happiness Scores (2015-2024)")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "happiness_histogram.png"))
    plt.clf()

    # Boxplot by year
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="year", y="happiness_score", data=df, palette="Set3")
    plt.title("Happiness Score Distributions by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.savefig(os.path.join(OUTPUT_DIR, "happiness_by_year.png"))
    plt.clf()

    # Scatter: GDP vs Happiness
    gdp_col = [
        col
        for col in df.columns
        if "gdp" in col.lower() or "economy" in col.lower()
    ]

    if gdp_col:
        actual_gdp_col = gdp_col[0]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=actual_gdp_col, y="happiness_score", data=df, alpha=0.7)
        plt.title("GDP per Capita vs Happiness Score")
        plt.xlabel(actual_gdp_col)
        plt.ylabel("Happiness Score")
        plt.savefig(os.path.join(OUTPUT_DIR, "gdp_vs_happiness.png"))
        plt.clf()

    # Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.clf()


# --- TASK 4: Hypothesis Testing ---

@task(name="Hypothesis Testing")
def run_hypothesis_tests(df):
    """Runs statistical tests on the happiness dataset and logs the results."""
    logger = get_run_logger()

    scores_2019 = df[df["year"] == 2019]["happiness_score"].dropna()
    scores_2020 = df[df["year"] == 2020]["happiness_score"].dropna()

    t_stat_1, p_val_1 = stats.ttest_ind(scores_2019, scores_2020)

    alpha = 0.05
    if p_val_1 < alpha:
        logger.info(
            f"Statistically significant difference in 2019 vs 2020 (p = {p_val_1:.4f})."
        )
    else:
        logger.info(
            f"No statistically significant difference in 2019 vs 2020 (p = {p_val_1:.4f})."
        )

    regional_means = (
        df.groupby("region")["happiness_score"].mean().sort_values(ascending=False)
    )
    top_region = regional_means.index[0]
    bottom_region = regional_means.index[-1]

    scores_top = df[df["region"] == top_region]["happiness_score"].dropna()
    scores_bottom = df[df["region"] == bottom_region]["happiness_score"].dropna()

    t_stat_2, p_val_2 = stats.ttest_ind(scores_top, scores_bottom)
    logger.info(
        f"Regional gap between {top_region} and {bottom_region}: p = {p_val_2:.10f}"
    )


# --- TASK 5: Correlation Analysis ---

@task(name="Correlation Analysis")
def run_correlation_analysis(df):
    """Computes correlations, applies Bonferroni correction, and logs significance."""
    logger = get_run_logger()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    explanatory_vars = [
        col for col in numeric_cols if col not in ["happiness_score", "year"]
    ]

    results = []
    for col in explanatory_vars:
        valid_data = df[["happiness_score", col]].dropna()
        if len(valid_data) > 1:
            r_val, p_val = stats.pearsonr(valid_data["happiness_score"], valid_data[col])
            results.append((col, r_val, p_val))

    num_tests = len(results)
    original_alpha = 0.05
    adjusted_alpha = original_alpha / num_tests if num_tests > 0 else 0.05

    for col, r_val, p_val in results:
        sig_adjusted = p_val < adjusted_alpha
        status = (
            "Significant (Bonferroni)" if sig_adjusted else "Not Significant"
        )
        logger.info(f"{col}: {status}")


# --- TASK 6: Summary Report ---

@task(name="Generate Summary Report")
def generate_summary_report(df):
    """Generates a human-readable summary report."""
    logger = get_run_logger()

    country_col = [col for col in df.columns if "country" in col.lower()][0]
    num_countries = df[country_col].nunique()
    num_years = df["year"].nunique()

    logger.info(f"1. DATASET SCOPE: {num_countries} countries over {num_years} years.")

    regional_means = (
        df.groupby("region")["happiness_score"].mean().sort_values(ascending=False)
    )
    logger.info(f"2. TOP REGION: {regional_means.index[0]}")
    logger.info(f"   BOTTOM REGION: {regional_means.index[-1]}")

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    explanatory_vars = [
        col for col in numeric_cols if col not in ["happiness_score", "year"]
    ]

    best_var = None
    max_r = 0
    adjusted_alpha = 0.05 / len(explanatory_vars) if len(explanatory_vars) > 0 else 0.05

    for col in explanatory_vars:
        valid_data = df[["happiness_score", col]].dropna()
        if len(valid_data) > 1:
            r_val, p_val = stats.pearsonr(valid_data["happiness_score"], valid_data[col])
            if p_val < adjusted_alpha and abs(r_val) > abs(max_r):
                max_r = r_val
                best_var = col

    if best_var:
        logger.info(
            f"3. STRONGEST DRIVER: '{best_var}' (r = {max_r:.4f}, passed Bonferroni threshold)."
        )
    else:
        logger.info(
            "3. STRONGEST DRIVER: No single variable passed the Bonferroni significance threshold."
        )


# --- MAIN PIPELINE FLOW ---

@flow(name="World Happiness Analysis Flow")
def happiness_pipeline():
    logger = get_run_logger()
    logger.info("Starting World Happiness Pipeline...")

    output_csv = os.path.join(OUTPUT_DIR, "merged_happiness.csv")

    merged_data = load_multiple_years(DATA_DIR, output_csv)
    calculate_descriptive_stats(merged_data)
    create_visualizations(merged_data)
    run_hypothesis_tests(merged_data)
    run_correlation_analysis(merged_data)
    generate_summary_report(merged_data)


if __name__ == "__main__":
    happiness_pipeline()
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Compute Cramer's V effect size for a contingency table."""
    if n == 0:
        return np.nan
    k = min(r - 1, c - 1)
    if k <= 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * k)))


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    df = pd.read_csv(base_dir / "boxes.csv")

    # Define key derived variables
    df["uses_social"] = df["y"].isin([2, 3]).astype(int)

    df_social = df[df["uses_social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Define coarse age groups to capture developmental stages
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    df_social["age_group"] = pd.cut(df_social["age"], bins=bins, labels=labels)

    results = {}

    # Overall distribution of outcomes
    outcome_counts = df["y"].value_counts().sort_index()
    results["outcome_counts"] = outcome_counts.to_dict()
    results["n"] = int(len(df))

    # Helper to run chi-square tests and effect sizes
    def chi_square_analysis(data, row_var, col_var, key_prefix):
        contingency = pd.crosstab(data[row_var], data[col_var])
        chi2, p, dof, _ = stats.chi2_contingency(contingency)
        v = cramers_v(chi2, contingency.to_numpy().sum(), *contingency.shape)
        results[f"{key_prefix}_contingency"] = contingency.to_dict()
        results[f"{key_prefix}_chi2"] = float(chi2)
        results[f"{key_prefix}_p"] = float(p)
        results[f"{key_prefix}_dof"] = int(dof)
        results[f"{key_prefix}_cramers_v"] = float(v)

    # 1. Reliance on social information vs culture
    chi_square_analysis(df, "uses_social", "culture", "uses_social_by_culture")

    # 2. Reliance on social information vs age group
    chi_square_analysis(df, "uses_social", "age_group", "uses_social_by_age_group")

    # 3. Majority preference vs culture (among children using social info)
    chi_square_analysis(df_social, "majority_choice", "culture", "majority_choice_by_culture")

    # 4. Majority preference vs age group (among children using social info)
    chi_square_analysis(df_social, "majority_choice", "age_group", "majority_choice_by_age_group")

    # Save intermediate detailed results for inspection if needed
    (base_dir / "analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


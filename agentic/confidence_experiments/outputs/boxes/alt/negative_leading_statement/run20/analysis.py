import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cramers_v(chi2: float, n: int, r: int, k: int) -> float:
    """Compute Cramér's V effect size for a contingency table."""
    if n <= 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * (min(r - 1, k - 1)))))


def chi2_test(table: pd.DataFrame):
    """Run chi-square test of independence and Cramér's V."""
    chi2, p, dof, _ = stats.chi2_contingency(table)
    r, k = table.shape
    n = int(table.to_numpy().sum())
    v = cramers_v(chi2, n, r, k)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": float(v),
        "n": n,
        "rows": int(r),
        "cols": int(k),
    }


def main():
    data_path = Path("boxes.csv")
    if not data_path.exists():
        raise FileNotFoundError("boxes.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Derived variables
    df["social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = np.where(
        df["y"] == 2,
        1,
        np.where(df["y"] == 3, 0, np.nan),
    )

    # Basic descriptives
    overall_social = df["social"].mean()
    demo_df = df[df["majority_choice"].notna()].copy()
    overall_majority = demo_df["majority_choice"].mean()

    # Contingency tables and tests
    social_culture_tab = pd.crosstab(df["culture"], df["social"])
    social_age_tab = pd.crosstab(df["age"], df["social"])

    majority_culture_tab = pd.crosstab(demo_df["culture"], demo_df["majority_choice"])
    majority_age_tab = pd.crosstab(demo_df["age"], demo_df["majority_choice"])

    results = {
        "n_total": int(len(df)),
        "overall_social_mean": float(overall_social),
        "overall_majority_mean": float(overall_majority),
        "social_by_culture": df.groupby("culture")["social"].mean().to_dict(),
        "social_by_age": df.groupby("age")["social"].mean().to_dict(),
        "majority_by_culture": demo_df.groupby("culture")["majority_choice"]
        .mean()
        .to_dict(),
        "majority_by_age": demo_df.groupby("age")["majority_choice"]
        .mean()
        .to_dict(),
        "tests": {
            "social_vs_culture": chi2_test(social_culture_tab),
            "social_vs_age": chi2_test(social_age_tab),
            "majority_vs_culture": chi2_test(majority_culture_tab),
            "majority_vs_age": chi2_test(majority_age_tab),
        },
    }

    # Pretty-print JSON results for manual inspection.
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


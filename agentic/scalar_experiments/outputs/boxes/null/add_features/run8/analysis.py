import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Basic sanity checks
    df = df.dropna(subset=["y", "age", "culture"])

    # Outcome coding
    # y: 1 = undemonstrated (no social info), 2 = majority, 3 = minority
    df["social_follow"] = np.where(df["y"] == 1, 0, 1)
    df["majority_choice"] = np.where(df["y"] == 2, 1, 0)

    # Restrict majority-preference analyses to trials that followed a demonstrated option
    df_demo = df[df["y"].isin([2, 3])].copy()

    # Construct coarse "developmental stage" groups using age quantiles
    # (younger / middle / older within this sample).
    df["age_group"] = pd.qcut(df["age"], q=3, labels=["young", "middle", "old"])
    df_demo["age_group"] = pd.qcut(df_demo["age"], q=3, labels=["young", "middle", "old"])

    results = {}

    # Descriptive statistics
    results["n_total"] = int(len(df))
    results["age_min"] = float(df["age"].min())
    results["age_max"] = float(df["age"].max())
    results["n_cultures"] = int(df["culture"].nunique())
    results["social_follow_rate"] = float(df["social_follow"].mean())
    results["majority_rate_overall"] = float(df["majority_choice"].mean())
    results["majority_rate_given_demo"] = float(df_demo["majority_choice"].mean())

    # Helper for chi-square tests
    def chi2_test_binary_vs_factor(binary, factor):
        table = pd.crosstab(binary, factor)
        chi2, p, dof, _ = stats.chi2_contingency(table)
        return {
            "chi2": float(chi2),
            "p": float(p),
            "dof": int(dof),
            "table": table.to_dict(),
        }

    # Reliance on social information: social_follow vs age_group and culture
    results["chi2_social_vs_age_group"] = chi2_test_binary_vs_factor(
        df["social_follow"], df["age_group"]
    )
    results["chi2_social_vs_culture"] = chi2_test_binary_vs_factor(
        df["social_follow"], df["culture"]
    )

    # Majority preference among those who followed any demonstration
    results["chi2_majority_vs_age_group"] = chi2_test_binary_vs_factor(
        df_demo["majority_choice"], df_demo["age_group"]
    )
    results["chi2_majority_vs_culture"] = chi2_test_binary_vs_factor(
        df_demo["majority_choice"], df_demo["culture"]
    )

    # Print a compact JSON summary to stdout for manual inspection.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


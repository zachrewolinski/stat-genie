import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def chi2_crosstab(x: pd.Series, y: pd.Series):
    """
    Run a chi-square test of independence and compute Cramer's V.
    Returns a small dict with key statistics and the contingency table.
    """
    table = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    if min(r - 1, k - 1) == 0:
        cramer_v = np.nan
    else:
        cramer_v = float(np.sqrt(chi2 / (n * (min(r - 1, k - 1)))))
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": cramer_v,
        "table": table,
    }


def summarize_proportions(df: pd.DataFrame, outcome: str, by: str):
    """
    Compute mean (proportion) of a binary outcome by grouping variable.
    """
    return (
        df.groupby(by)[outcome]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop", "count": "n"})
        .sort_index()
    )


def main():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Core variables based on metadata:
    # y: 1 = undemonstrated option, 2 = majority option, 3 = minority option
    # culture: site ID
    # age: age group
    # Derive binary indicators for analysis.
    df = df.dropna(subset=["y", "age", "culture"])

    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["social_choice"] = (df["y"] != 1).astype(int)

    # Ensure categorical treatment for chi-square tests.
    df["age_cat"] = df["age"].astype("category")
    df["culture_cat"] = df["culture"].astype("category")

    results = {}

    # Majority vs. culture and age
    results["majority_by_culture"] = chi2_crosstab(
        df["culture_cat"], df["majority_choice"]
    )
    results["majority_by_age"] = chi2_crosstab(df["age_cat"], df["majority_choice"])

    # Social (any demonstrated option) vs. culture and age
    results["social_by_culture"] = chi2_crosstab(
        df["culture_cat"], df["social_choice"]
    )
    results["social_by_age"] = chi2_crosstab(df["age_cat"], df["social_choice"])

    # Proportion summaries to interpret direction and patterns.
    summaries = {
        "majority_prop_by_culture": summarize_proportions(
            df, "majority_choice", "culture_cat"
        ).to_dict(orient="index"),
        "majority_prop_by_age": summarize_proportions(
            df, "majority_choice", "age_cat"
        ).to_dict(orient="index"),
        "social_prop_by_culture": summarize_proportions(
            df, "social_choice", "culture_cat"
        ).to_dict(orient="index"),
        "social_prop_by_age": summarize_proportions(
            df, "social_choice", "age_cat"
        ).to_dict(orient="index"),
    }

    # Print a compact JSON summary of key statistics for inspection.
    output = {
        "tests": {
            name: {
                "chi2": res["chi2"],
                "p_value": res["p_value"],
                "dof": res["dof"],
                "cramers_v": res["cramers_v"],
            }
            for name, res in results.items()
        },
        "summaries": summaries,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()


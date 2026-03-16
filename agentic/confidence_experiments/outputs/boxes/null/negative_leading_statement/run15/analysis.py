import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Derive key behavioral constructs
    df["social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)
    # Restrict to trials where a social option was chosen for majority-bias analyses
    df["social_trial"] = df["y"].isin([2, 3]).astype(int)
    return df


def fit_logistic(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data)
    result = model.fit(disp=False)
    return result


def analyze(df: pd.DataFrame):
    results = {}

    # 1. Reliance on social information (choosing any demonstrated option vs undemonstrated)
    social_model = fit_logistic(
        "social ~ age + C(culture) + gender + majority_first",
        df,
    )
    results["social_reliance"] = {
        "n": int(df.shape[0]),
        "coef": social_model.params.to_dict(),
        "pvalues": social_model.pvalues.to_dict(),
    }

    # 2. Majority preference among social choices only
    social_df = df[df["social_trial"] == 1].copy()
    majority_model = fit_logistic(
        "majority_choice ~ age + C(culture) + gender + majority_first",
        social_df,
    )
    results["majority_preference"] = {
        "n": int(social_df.shape[0]),
        "coef": majority_model.params.to_dict(),
        "pvalues": majority_model.pvalues.to_dict(),
    }

    # 3. Descriptive summaries by culture and age (for interpretation)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        right=True,
    )

    desc = {}
    # Social reliance: P(social) by culture and age_group
    desc["social_by_culture"] = (
        df.groupby("culture")["social"].mean().to_dict()
    )
    desc["social_by_age_group"] = (
        df.groupby("age_group")["social"].mean().to_dict()
    )

    # Majority preference: P(majority_choice | social) by culture and age_group
    desc["majority_by_culture"] = (
        social_df.groupby("culture")["majority_choice"].mean().to_dict()
    )
    social_df["age_group"] = pd.cut(
        social_df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        right=True,
    )
    desc["majority_by_age_group"] = (
        social_df.groupby("age_group")["majority_choice"].mean().to_dict()
    )

    results["descriptives"] = desc
    return results


def main():
    csv_path = Path("boxes.csv")
    df = load_data(csv_path)
    results = analyze(df)

    # Print a compact JSON summary for the agent to inspect
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()


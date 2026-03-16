import json
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


@dataclass
class EffectTestResult:
    lr_stat: float
    df_diff: int
    p_value: float


def likelihood_ratio_test(
    full_formula: str,
    reduced_formula: str,
    data: pd.DataFrame,
) -> EffectTestResult:
    """Run a likelihood-ratio test between nested logistic models."""
    full_model = smf.logit(full_formula, data=data).fit(disp=False)
    reduced_model = smf.logit(reduced_formula, data=data).fit(disp=False)

    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)

    return EffectTestResult(lr_stat=lr_stat, df_diff=int(df_diff), p_value=float(p_value))


def compute_site_summaries(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("site")
        .agg(
            n=("outcome", "size"),
            social_rate=("social", "mean"),
            majority_rate=("maj_choice", "mean"),
            mean_age=("age", "mean"),
        )
        .reset_index()
    )
    return grouped


def compute_age_summaries(df: pd.DataFrame) -> pd.DataFrame:
    df_age = (
        df.groupby("age")
        .agg(
            n=("outcome", "size"),
            social_rate=("social", "mean"),
            majority_rate=("maj_choice", "mean"),
        )
        .reset_index()
    )
    return df_age


def main() -> None:
    df = pd.read_csv("boxes.csv")

    df = df.rename(
        columns={
            "feature1": "outcome",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "maj_first",
            "feature5": "site",
        }
    )

    # Binary indicators for the two key constructs
    df["social"] = (df["outcome"] != 1).astype(int)
    df["maj_choice"] = np.where(
        df["outcome"] == 1,
        np.nan,
        np.where(df["outcome"] == 2, 1, 0),
    )

    # Treat site and gender as categorical predictors
    df["site"] = df["site"].astype("category")
    df["gender"] = df["gender"].astype("category")

    # Descriptive summaries
    site_summary = compute_site_summaries(df)
    age_summary = compute_age_summaries(df)

    print("=== Site-level summaries ===")
    print(site_summary.to_string(index=False))
    print("\n=== Age-level summaries ===")
    print(age_summary.to_string(index=False))

    # Logistic regression: reliance on social information
    print("\n=== Logistic regression: social information use ===")
    social_full = "social ~ age + I(age ** 2) + C(site) + C(gender) + maj_first"
    social_age_reduced = "social ~ C(site) + C(gender) + maj_first"
    social_site_reduced = "social ~ age + I(age ** 2) + C(gender) + maj_first"

    social_age_effect = likelihood_ratio_test(
        full_formula=social_full,
        reduced_formula=social_age_reduced,
        data=df,
    )
    social_site_effect = likelihood_ratio_test(
        full_formula=social_full,
        reduced_formula=social_site_reduced,
        data=df,
    )

    print("Age effect on social use (LR test):", asdict(social_age_effect))
    print("Site effect on social use (LR test):", asdict(social_site_effect))

    # Logistic regression: majority preference among social choices
    df_social = df[df["social"] == 1].copy()

    print("\nNumber of social choices:", len(df_social), "of", len(df))

    print("\n=== Logistic regression: majority vs. minority among social choices ===")
    majority_full = "maj_choice ~ age + I(age ** 2) + C(site) + C(gender) + maj_first"
    majority_age_reduced = "maj_choice ~ C(site) + C(gender) + maj_first"
    majority_site_reduced = "maj_choice ~ age + I(age ** 2) + C(gender) + maj_first"

    majority_age_effect = likelihood_ratio_test(
        full_formula=majority_full,
        reduced_formula=majority_age_reduced,
        data=df_social,
    )
    majority_site_effect = likelihood_ratio_test(
        full_formula=majority_full,
        reduced_formula=majority_site_reduced,
        data=df_social,
    )

    print("Age effect on majority preference (LR test):", asdict(majority_age_effect))
    print("Site effect on majority preference (LR test):", asdict(majority_site_effect))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Rename for clarity
    df = df.rename(
        columns={
            "feature1": "outcome",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Derived variables
    df["social_reliance"] = np.where(df["outcome"].isin([2, 3]), 1, 0)

    social_df = df[df["outcome"].isin([2, 3])].copy()
    social_df["majority_choice"] = np.where(social_df["outcome"] == 2, 1, 0)

    # Basic summaries
    overall_social_rate = df["social_reliance"].mean()
    overall_majority_rate = social_df["majority_choice"].mean()

    # Chi-squared tests for variation across sites (cultures)
    site_social_table = pd.crosstab(df["site"], df["social_reliance"])
    chi2_social, p_social, dof_social, _ = stats.chi2_contingency(site_social_table)

    site_majority_table = pd.crosstab(social_df["site"], social_df["majority_choice"])
    chi2_majority, p_majority, dof_majority, _ = stats.chi2_contingency(
        site_majority_table
    )

    # Logistic regressions for age effects (developmental change)
    # Model 1: reliance on social information
    model_social = smf.logit(
        "social_reliance ~ age + C(site) + C(gender) + majority_first", data=df
    ).fit(disp=False)
    age_coef_social = model_social.params["age"]
    age_p_social = model_social.pvalues["age"]

    # Model 2: preference for majority over minority among social learners
    model_majority = smf.logit(
        "majority_choice ~ age + C(site) + C(gender) + majority_first", data=social_df
    ).fit(disp=False)
    age_coef_majority = model_majority.params["age"]
    age_p_majority = model_majority.pvalues["age"]

    # Simple age-bin summaries (for interpretability)
    bins = [4, 6, 8, 10, 12, 14.1]
    labels = ["4-5", "6-7", "8-9", "10-11", "12-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    social_df["age_group"] = pd.cut(social_df["age"], bins=bins, labels=labels, right=False)

    age_group_social = (
        df.groupby("age_group")["social_reliance"].mean().to_dict()
    )
    age_group_majority = (
        social_df.groupby("age_group")["majority_choice"].mean().to_dict()
    )

    results = {
        "overall_social_rate": float(overall_social_rate),
        "overall_majority_rate": float(overall_majority_rate),
        "chi2_social": {
            "chi2": float(chi2_social),
            "p": float(p_social),
            "dof": int(dof_social),
        },
        "chi2_majority": {
            "chi2": float(chi2_majority),
            "p": float(p_majority),
            "dof": int(dof_majority),
        },
        "logit_social_age": {
            "coef": float(age_coef_social),
            "p": float(age_p_social),
        },
        "logit_majority_age": {
            "coef": float(age_coef_majority),
            "p": float(age_p_majority),
        },
        "age_group_social_rates": age_group_social,
        "age_group_majority_rates": age_group_majority,
    }

    # Save numeric results so they can be inspected when forming the conclusion.
    Path("analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


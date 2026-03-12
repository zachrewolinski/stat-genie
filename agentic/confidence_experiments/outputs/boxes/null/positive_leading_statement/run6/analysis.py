import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def likelihood_ratio_test(full_result, reduced_result, df_full, df_reduced):
    """Compute likelihood-ratio test between two nested models."""
    lr_stat = 2 * (full_result.llf - reduced_result.llf)
    df_diff = df_full - df_reduced
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return float(lr_stat), int(df_diff), float(p_value)


def main():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Encode key behavioural outcomes
    df["social"] = (df["y"] != 1).astype(int)  # 1 = undemonstrated option
    df["majority_choice"] = (df["y"] == 2).astype(int)  # 2 = majority option

    # Basic descriptives
    n = len(df)
    overall_social = df["social"].mean()
    overall_majority = df["majority_choice"].mean()

    # Social information use: logistic regression social ~ age + culture
    model_social_full = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False)
    model_social_no_culture = smf.logit("social ~ age", data=df).fit(disp=False)
    model_social_no_age = smf.logit("social ~ C(culture)", data=df).fit(disp=False)

    lr_social_culture = likelihood_ratio_test(
        model_social_full, model_social_no_culture, df_full=len(model_social_full.params), df_reduced=len(model_social_no_culture.params)
    )
    lr_social_age = likelihood_ratio_test(
        model_social_full, model_social_no_age, df_full=len(model_social_full.params), df_reduced=len(model_social_no_age.params)
    )

    # Majority preference among all children: majority_choice ~ age + culture
    model_maj_full = smf.logit("majority_choice ~ age + C(culture)", data=df).fit(disp=False)
    model_maj_no_culture = smf.logit("majority_choice ~ age", data=df).fit(disp=False)
    model_maj_no_age = smf.logit("majority_choice ~ C(culture)", data=df).fit(disp=False)

    lr_maj_culture = likelihood_ratio_test(
        model_maj_full, model_maj_no_culture, df_full=len(model_maj_full.params), df_reduced=len(model_maj_no_culture.params)
    )
    lr_maj_age = likelihood_ratio_test(
        model_maj_full, model_maj_no_age, df_full=len(model_maj_full.params), df_reduced=len(model_maj_no_age.params)
    )

    # Age-grouped descriptives for interpretability
    df["age_group"] = pd.cut(df["age"], bins=[3, 6, 9, 12, 15], labels=["4-6", "7-9", "10-12", "13-14"])
    social_by_age = df.groupby("age_group")["social"].mean().to_dict()
    majority_by_age = df.groupby("age_group")["majority_choice"].mean().to_dict()

    social_by_culture = df.groupby("culture")["social"].mean().to_dict()
    majority_by_culture = df.groupby("culture")["majority_choice"].mean().to_dict()

    results = {
        "n": int(n),
        "overall": {
            "social_mean": float(overall_social),
            "majority_mean": float(overall_majority),
        },
        "lr_tests": {
            "social_culture": {
                "lr_stat": lr_social_culture[0],
                "df_diff": lr_social_culture[1],
                "p_value": lr_social_culture[2],
            },
            "social_age": {
                "lr_stat": lr_social_age[0],
                "df_diff": lr_social_age[1],
                "p_value": lr_social_age[2],
            },
            "majority_culture": {
                "lr_stat": lr_maj_culture[0],
                "df_diff": lr_maj_culture[1],
                "p_value": lr_maj_culture[2],
            },
            "majority_age": {
                "lr_stat": lr_maj_age[0],
                "df_diff": lr_maj_age[1],
                "p_value": lr_maj_age[2],
            },
        },
        "descriptives": {
            "social_by_age": social_by_age,
            "majority_by_age": majority_by_age,
            "social_by_culture": social_by_culture,
            "majority_by_culture": majority_by_culture,
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()


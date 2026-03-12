import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def likelihood_ratio_test(full_model, reduced_model, name: str):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return {
        "effect": name,
        "lr_stat": float(lr_stat),
        "df": int(df_diff),
        "p_value": float(p_value),
    }


def main():
    base_path = Path(__file__).parent
    csv_path = base_path / "boxes.csv"
    info_path = base_path / "info.json"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    df = pd.read_csv(csv_path)

    # Derived variables
    df["social_reliance"] = df["feature1"].isin([2, 3]).astype(int)
    df["used_social"] = df["feature1"].isin([2, 3])
    df["majority_choice"] = (df["feature1"] == 2).astype(int)

    # Basic descriptives
    overall_social = df["social_reliance"].mean()
    overall_majority = df.loc[df["used_social"], "majority_choice"].mean()

    print("Overall proportion using social information:", overall_social)
    print(
        "Overall proportion choosing majority (conditional on using social info):",
        overall_majority,
    )

    # Treat gender and site as categorical, majority-first as binary covariate, age as continuous
    df["feature2"] = df["feature2"].astype("category")
    df["feature5"] = df["feature5"].astype("category")

    # Model 1: reliance on social information (any demonstrated option vs undemonstrated)
    formula_full_social = "social_reliance ~ feature3 + C(feature5) + C(feature2) + feature4"
    formula_no_site_social = "social_reliance ~ feature3 + C(feature2) + feature4"
    formula_no_age_social = "social_reliance ~ C(feature5) + C(feature2) + feature4"

    model_social_full = smf.glm(
        formula_full_social, data=df, family=sm.families.Binomial()
    ).fit()
    model_social_no_site = smf.glm(
        formula_no_site_social, data=df, family=sm.families.Binomial()
    ).fit()
    model_social_no_age = smf.glm(
        formula_no_age_social, data=df, family=sm.families.Binomial()
    ).fit()

    lr_site_social = likelihood_ratio_test(
        model_social_full, model_social_no_site, "site_on_social_reliance"
    )
    lr_age_social = likelihood_ratio_test(
        model_social_full, model_social_no_age, "age_on_social_reliance"
    )

    print("\nModel 1: Social reliance (any demonstrated option)")
    print(model_social_full.summary())
    print("LRT for site effect on social reliance:", lr_site_social)
    print("LRT for age effect on social reliance:", lr_age_social)

    # Model 2: majority preference among children who used social information
    df_social = df[df["used_social"]].copy()

    formula_full_majority = (
        "majority_choice ~ feature3 + C(feature5) + C(feature2) + feature4"
    )
    formula_no_site_majority = "majority_choice ~ feature3 + C(feature2) + feature4"
    formula_no_age_majority = "majority_choice ~ C(feature5) + C(feature2) + feature4"

    model_majority_full = smf.glm(
        formula_full_majority, data=df_social, family=sm.families.Binomial()
    ).fit()
    model_majority_no_site = smf.glm(
        formula_no_site_majority, data=df_social, family=sm.families.Binomial()
    ).fit()
    model_majority_no_age = smf.glm(
        formula_no_age_majority, data=df_social, family=sm.families.Binomial()
    ).fit()

    lr_site_majority = likelihood_ratio_test(
        model_majority_full, model_majority_no_site, "site_on_majority_preference"
    )
    lr_age_majority = likelihood_ratio_test(
        model_majority_full, model_majority_no_age, "age_on_majority_preference"
    )

    print("\nModel 2: Majority preference (among social learners)")
    print(model_majority_full.summary())
    print("LRT for site effect on majority preference:", lr_site_majority)
    print("LRT for age effect on majority preference:", lr_age_majority)

    # Store key numeric results to help interpretation if needed
    results = {
        "overall_social_reliance": float(overall_social),
        "overall_majority_preference": float(overall_majority),
        "lr_tests": {
            "social_reliance": {
                "site": lr_site_social,
                "age": lr_age_social,
            },
            "majority_preference": {
                "site": lr_site_majority,
                "age": lr_age_majority,
            },
        },
    }

    # Save machine-readable results for potential downstream use
    (base_path / "analysis_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

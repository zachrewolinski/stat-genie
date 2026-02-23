import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Outcome: 1 = undemonstrated option, 2 = majority option, 3 = minority option
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    # Descriptive statistics
    overall_rate = df["majority_choice"].mean()
    by_age = df.groupby("age")["majority_choice"].agg(["mean", "count"]).reset_index()
    by_site = df.groupby("y")["majority_choice"].agg(["mean", "count"]).reset_index()

    print("Overall majority-choice rate:", overall_rate)
    print("\nMajority-choice rate by age:")
    print(by_age)
    print("\nMajority-choice rate by site (y = site ID):")
    print(by_site)

    # Effect sizes
    age_effect = by_age["mean"].max() - by_age["mean"].min()
    site_effect = by_site["mean"].max() - by_site["mean"].min()
    print("\nApproximate effect size (absolute difference in proportions):")
    print(f"  Age effect (min->max age-specific rate): {age_effect:.3f}")
    print(f"  Site effect (min->max site-specific rate): {site_effect:.3f}")

    # Logistic regression using GLM with binomial family
    model_age = smf.glm(
        "majority_choice ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    print("\nGLM (Binomial) majority_choice ~ age")
    print(model_age.summary())

    model_age_site = smf.glm(
        "majority_choice ~ age + C(y)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    print("\nGLM (Binomial) majority_choice ~ age + C(y)")
    print(model_age_site.summary())

    # Extract key p-values
    p_age_only = float(model_age.pvalues["age"])
    p_age_site = float(model_age_site.pvalues["age"])

    site_params = {
        name: float(p)
        for name, p in model_age_site.pvalues.items()
        if name.startswith("C(y)[T.")
    }

    if site_params:
        min_site_p = float(min(site_params.values()))
        max_site_p = float(max(site_params.values()))
    else:
        min_site_p = np.nan
        max_site_p = np.nan

    print("\nKey p-values:")
    print(f"  Age effect (age-only model): p = {p_age_only:.4g}")
    print(f"  Age effect (controlling for site): p = {p_age_site:.4g}")
    print("  Site (culture proxy) dummy p-values:")
    for name, p in site_params.items():
        print(f"    {name}: p = {p:.4g}")
    print(
        f"  Min site p-value: {min_site_p:.4g}, "
        f"max site p-value: {max_site_p:.4g}"
    )

    # Chi-square tests with grouped ages and binary majority-choice outcome
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=True,
    )

    ct_age = pd.crosstab(df["age_group"], df["majority_choice"])
    chi2_age, p_age_chi2, dof_age, _ = stats.chi2_contingency(ct_age)

    ct_site = pd.crosstab(df["y"], df["majority_choice"])
    chi2_site, p_site_chi2, dof_site, _ = stats.chi2_contingency(ct_site)

    print("\nChi-square tests (majority_choice vs predictors):")
    print("  Age group vs majority_choice:")
    print(ct_age)
    print(
        f"    chi2 = {chi2_age:.3f}, dof = {dof_age}, "
        f"p = {p_age_chi2:.4g}"
    )

    print("\n  Site (y) vs majority_choice:")
    print(ct_site)
    print(
        f"    chi2 = {chi2_site:.3f}, dof = {dof_site}, "
        f"p = {p_site_chi2:.4g}"
    )

    # Save a compact machine-readable summary to help with interpretation if needed
    summary = {
        "overall_majority_rate": float(overall_rate),
        "age_effect_diff": float(age_effect),
        "site_effect_diff": float(site_effect),
        "glm_age_p": p_age_only,
        "glm_age_site_p": p_age_site,
        "glm_site_p_values": site_params,
        "chi2_age_p": float(p_age_chi2),
        "chi2_site_p": float(p_site_chi2),
    }

    with open("analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()


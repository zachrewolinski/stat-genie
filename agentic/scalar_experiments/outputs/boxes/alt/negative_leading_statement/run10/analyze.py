import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def likelihood_ratio_test(full_model, reduced_model, df_diff):
    """Return LR statistic and p-value comparing two nested models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, p_value


def analyze_social_reliance(df: pd.DataFrame):
    """
    Binary outcome: relied on social information (majority or minority)
    vs undemonstrated option.
    """
    df = df.copy()
    df["social"] = (df["y"].isin([2, 3])).astype(int)

    # Descriptive stats
    overall_rate = df["social"].mean()
    by_culture = df.groupby("culture")["social"].mean()

    # Logistic regression with age and culture as predictors
    model_full = smf.glm(
        formula="social ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # Reduced models for LR tests
    model_age_only = smf.glm(
        formula="social ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    model_culture_only = smf.glm(
        formula="social ~ C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    model_intercept_only = smf.glm(
        formula="social ~ 1",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # LR tests
    lr_age_vs_intercept, p_age_vs_intercept = likelihood_ratio_test(
        model_age_only, model_intercept_only, df_diff=1
    )
    lr_culture_vs_intercept, p_culture_vs_intercept = likelihood_ratio_test(
        model_culture_only, model_intercept_only, df_diff=model_culture_only.df_model
    )
    lr_full_vs_age, p_full_vs_age = likelihood_ratio_test(
        model_full, model_age_only, df_diff=model_full.df_model - model_age_only.df_model
    )
    lr_full_vs_culture, p_full_vs_culture = likelihood_ratio_test(
        model_full,
        model_culture_only,
        df_diff=model_full.df_model - model_culture_only.df_model,
    )

    return {
        "overall_social_rate": float(overall_rate),
        "social_rate_by_culture": by_culture.to_dict(),
        "model_full_summary": model_full.summary().as_text(),
        "lr_tests": {
            "age_vs_intercept": {"lr": float(lr_age_vs_intercept), "p": float(p_age_vs_intercept)},
            "culture_vs_intercept": {
                "lr": float(lr_culture_vs_intercept),
                "p": float(p_culture_vs_intercept),
            },
            "full_vs_age": {"lr": float(lr_full_vs_age), "p": float(p_full_vs_age)},
            "full_vs_culture": {"lr": float(lr_full_vs_culture), "p": float(p_full_vs_culture)},
        },
    }


def analyze_majority_preference(df: pd.DataFrame):
    """
    Among children who followed social information (majority or minority),
    model preference for the majority option vs minority.
    """
    df = df[df["y"].isin([2, 3])].copy()
    df["majority_choice"] = (df["y"] == 2).astype(int)

    if df.empty:
        raise ValueError("No trials where children followed social information.")

    overall_rate = df["majority_choice"].mean()
    by_culture = df.groupby("culture")["majority_choice"].mean()

    model_full = smf.glm(
        formula="majority_choice ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    model_age_only = smf.glm(
        formula="majority_choice ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    model_culture_only = smf.glm(
        formula="majority_choice ~ C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    model_intercept_only = smf.glm(
        formula="majority_choice ~ 1",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    lr_age_vs_intercept, p_age_vs_intercept = likelihood_ratio_test(
        model_age_only, model_intercept_only, df_diff=1
    )
    lr_culture_vs_intercept, p_culture_vs_intercept = likelihood_ratio_test(
        model_culture_only, model_intercept_only, df_diff=model_culture_only.df_model
    )
    lr_full_vs_age, p_full_vs_age = likelihood_ratio_test(
        model_full, model_age_only, df_diff=model_full.df_model - model_age_only.df_model
    )
    lr_full_vs_culture, p_full_vs_culture = likelihood_ratio_test(
        model_full,
        model_culture_only,
        df_diff=model_full.df_model - model_culture_only.df_model,
    )

    return {
        "overall_majority_rate": float(overall_rate),
        "majority_rate_by_culture": by_culture.to_dict(),
        "model_full_summary": model_full.summary().as_text(),
        "lr_tests": {
            "age_vs_intercept": {"lr": float(lr_age_vs_intercept), "p": float(p_age_vs_intercept)},
            "culture_vs_intercept": {
                "lr": float(lr_culture_vs_intercept),
                "p": float(p_culture_vs_intercept),
            },
            "full_vs_age": {"lr": float(lr_full_vs_age), "p": float(p_full_vs_age)},
            "full_vs_culture": {"lr": float(lr_full_vs_culture), "p": float(p_full_vs_culture)},
        },
    }


def main():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Encode culture as categorical
    df["culture"] = df["culture"].astype(int).astype("category")

    social_results = analyze_social_reliance(df)
    majority_results = analyze_majority_preference(df)

    results = {
        "social_reliance": social_results,
        "majority_preference": majority_results,
    }

    # Save detailed numeric results to a JSON file for inspection if needed.
    out_path = Path("analysis_results.json")
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Also print key statistics to stdout for quick inspection.
    print("Social reliance overall rate:", social_results["overall_social_rate"])
    print("Social reliance LR tests:", social_results["lr_tests"])
    print("Majority preference overall rate:", majority_results["overall_majority_rate"])
    print("Majority preference LR tests:", majority_results["lr_tests"])


if __name__ == "__main__":
    main()


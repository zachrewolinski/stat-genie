import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )
    df["site"] = df["site"].astype("category")

    # Social reliance: choosing any demonstrated option vs undemonstrated option.
    df["social_reliance"] = (df["choice"] != 1).astype(int)

    # Majority preference: among children who used social information.
    social_df = df[df["choice"].isin([2, 3])].copy()
    social_df["majority_choice"] = (social_df["choice"] == 2).astype(int)

    results = {}

    # Descriptive statistics for context.
    choice_props = df["choice"].value_counts(normalize=True).sort_index()
    results["overall_choice_proportions"] = {
        int(k): float(v) for k, v in choice_props.items()
    }

    # Age grouped descriptives.
    age_bins = [4, 6, 8, 10, 12, 14]
    age_labels = ["4-6", "6-8", "8-10", "10-12", "12-14"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, include_lowest=True, right=False)
    social_df["age_group"] = pd.cut(
        social_df["age"], bins=age_bins, labels=age_labels, include_lowest=True, right=False
    )

    age_group_social = (
        df.groupby("age_group")["social_reliance"].mean().dropna().to_dict()
    )
    age_group_majority = (
        social_df.groupby("age_group")["majority_choice"].mean().dropna().to_dict()
    )

    results["age_group_social_reliance_mean"] = {
        str(k): float(v) for k, v in age_group_social.items()
    }
    results["age_group_majority_preference_mean"] = {
        str(k): float(v) for k, v in age_group_majority.items()
    }

    # Descriptives by site.
    site_social = df.groupby("site")["social_reliance"].mean().to_dict()
    site_majority = social_df.groupby("site")["majority_choice"].mean().to_dict()
    results["site_social_reliance_mean"] = {
        str(k): float(v) for k, v in site_social.items()
    }
    results["site_majority_preference_mean"] = {
        str(k): float(v) for k, v in site_majority.items()
    }

    # Logistic regression: social_reliance ~ age + site.
    model_social_age = smf.logit("social_reliance ~ age", data=df).fit(disp=False)
    model_social_age_site = smf.logit("social_reliance ~ age + C(site)", data=df).fit(
        disp=False
    )

    # Likelihood ratio test for site effects on social_reliance.
    lr_stat_site_social = 2 * (
        model_social_age_site.llf - model_social_age.llf
    )
    df_diff_site_social = (
        model_social_age_site.df_model - model_social_age.df_model
    )
    p_site_social = stats.chi2.sf(lr_stat_site_social, df_diff_site_social)

    # Age effect on social_reliance: compare full model vs intercept-only.
    model_social_intercept = smf.logit("social_reliance ~ 1", data=df).fit(disp=False)
    lr_stat_age_social = 2 * (
        model_social_age_site.llf - model_social_intercept.llf
    )
    df_diff_age_social = (
        model_social_age_site.df_model - model_social_intercept.df_model
    )
    p_age_social = stats.chi2.sf(lr_stat_age_social, df_diff_age_social)

    results["social_reliance_lr_tests"] = {
        "lr_stat_site": float(lr_stat_site_social),
        "df_site": int(df_diff_site_social),
        "p_value_site": float(p_site_social),
        "lr_stat_age_overall": float(lr_stat_age_social),
        "df_age_overall": int(df_diff_age_social),
        "p_value_age_overall": float(p_age_social),
        "coef_age": float(model_social_age_site.params["age"]),
    }

    # Logistic regression: majority_choice ~ age + site (only among social learners).
    model_maj_age = smf.logit("majority_choice ~ age", data=social_df).fit(disp=False)
    model_maj_age_site = smf.logit(
        "majority_choice ~ age + C(site)", data=social_df
    ).fit(disp=False)

    lr_stat_site_majority = 2 * (
        model_maj_age_site.llf - model_maj_age.llf
    )
    df_diff_site_majority = (
        model_maj_age_site.df_model - model_maj_age.df_model
    )
    p_site_majority = stats.chi2.sf(lr_stat_site_majority, df_diff_site_majority)

    model_maj_intercept = smf.logit("majority_choice ~ 1", data=social_df).fit(
        disp=False
    )
    lr_stat_age_majority = 2 * (
        model_maj_age_site.llf - model_maj_intercept.llf
    )
    df_diff_age_majority = (
        model_maj_age_site.df_model - model_maj_intercept.df_model
    )
    p_age_majority = stats.chi2.sf(lr_stat_age_majority, df_diff_age_majority)

    results["majority_preference_lr_tests"] = {
        "lr_stat_site": float(lr_stat_site_majority),
        "df_site": int(df_diff_site_majority),
        "p_value_site": float(p_site_majority),
        "lr_stat_age_overall": float(lr_stat_age_majority),
        "df_age_overall": int(df_diff_age_majority),
        "p_value_age_overall": float(p_age_majority),
        "coef_age": float(model_maj_age_site.params["age"]),
    }

    # Save intermediate results so they can be inspected from the shell.
    Path("analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


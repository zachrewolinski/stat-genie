import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def load_data():
    df = pd.read_csv("boxes.csv")
    # Outcome coding:
    # 1 = undemonstrated third option (asocial choice)
    # 2 = majority option
    # 3 = minority option
    df["social_use"] = df["feature1"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["feature1"] == 2).astype(int)
    df["majority_given_social"] = np.where(
        df["social_use"] == 1, df["majority_choice"], np.nan
    )

    df["age"] = df["feature3"].astype(float)
    df["gender"] = df["feature2"].astype(int)
    df["majority_first"] = df["feature4"].astype(int)
    # Treat site as categorical to capture cultural context
    df["site"] = df["feature5"].astype("category")
    return df


def logistic_lr_test(full_model, reduced_model):
    """Likelihood-ratio test comparing nested logistic models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def summarize_social_use(df):
    overall_social = df["social_use"].mean()
    by_site = df.groupby("site")["social_use"].mean()
    print("Overall proportion using social information:", round(overall_social, 3))
    print("Proportion using social information by site:")
    print(by_site.round(3))


def summarize_majority_choice(df):
    overall_majority = df["majority_choice"].mean()
    by_site = df.groupby("site")["majority_choice"].mean()
    print("Overall proportion choosing majority option:", round(overall_majority, 3))
    print("Proportion choosing majority option by site:")
    print(by_site.round(3))


def fit_models(df):
    results = {}

    # Logistic regression: reliance on social information
    social_full = smf.logit(
        "social_use ~ age + C(site) + gender + majority_first", data=df
    ).fit(disp=False)
    social_no_site = smf.logit(
        "social_use ~ age + gender + majority_first", data=df
    ).fit(disp=False)
    social_no_age = smf.logit(
        "social_use ~ C(site) + gender + majority_first", data=df
    ).fit(disp=False)

    lr_site_social = logistic_lr_test(social_full, social_no_site)
    lr_age_social = logistic_lr_test(social_full, social_no_age)
    p_age_coef_social = social_full.pvalues.get("age", np.nan)

    results["social"] = {
        "model": social_full,
        "lr_site": lr_site_social,
        "lr_age": lr_age_social,
        "p_age_coef": float(p_age_coef_social),
    }

    # Logistic regression: preference for majority option (overall)
    majority_full = smf.logit(
        "majority_choice ~ age + C(site) + gender + majority_first", data=df
    ).fit(disp=False)
    majority_no_site = smf.logit(
        "majority_choice ~ age + gender + majority_first", data=df
    ).fit(disp=False)
    majority_no_age = smf.logit(
        "majority_choice ~ C(site) + gender + majority_first", data=df
    ).fit(disp=False)

    lr_site_majority = logistic_lr_test(majority_full, majority_no_site)
    lr_age_majority = logistic_lr_test(majority_full, majority_no_age)
    p_age_coef_majority = majority_full.pvalues.get("age", np.nan)

    results["majority"] = {
        "model": majority_full,
        "lr_site": lr_site_majority,
        "lr_age": lr_age_majority,
        "p_age_coef": float(p_age_coef_majority),
    }

    return results


def estimate_effects(df, model, outcome_label):
    """Estimate age and site effects via predicted probabilities."""
    # Age effect: compare younger vs older ages while holding other variables at observed values
    df_young = df.copy()
    df_old = df.copy()
    df_young["age"] = 5.0
    df_old["age"] = 12.0
    prob_young = model.predict(df_young).mean()
    prob_old = model.predict(df_old).mean()

    print(
        f"\nEstimated {outcome_label} probability at age 5:  {prob_young:.3f}",
        f"\nEstimated {outcome_label} probability at age 12: {prob_old:.3f}",
        f"\nDifference (12 - 5 years): {prob_old - prob_young:.3f}",
    )

    # Site effects: predicted probabilities by site, averaging over other covariates
    sites = df["site"].cat.categories
    site_probs = {}
    base_df = df.copy()
    for s in sites:
        df_site = base_df.copy()
        df_site["site"] = s
        site_probs[str(s)] = float(model.predict(df_site).mean())

    print(f"\nPredicted {outcome_label} probabilities by site:")
    for s, p in site_probs.items():
        print(f"  Site {s}: {p:.3f}")

    return {
        "prob_young": float(prob_young),
        "prob_old": float(prob_old),
        "prob_diff": float(prob_old - prob_young),
        "site_probs": site_probs,
    }


def main():
    df = load_data()

    print("Number of observations:", len(df))
    summarize_social_use(df)
    summarize_majority_choice(df)

    results = fit_models(df)

    print("\n=== Reliance on social information ===")
    lr_stat, df_diff, p_site_social = results["social"]["lr_site"]
    print(
        f"LR test for site effect (social_use): "
        f"chi2 = {lr_stat:.2f}, df = {df_diff}, p = {p_site_social:.4g}"
    )
    lr_stat_age, df_diff_age, p_age_social = results["social"]["lr_age"]
    print(
        f"LR test for age effect (social_use): "
        f"chi2 = {lr_stat_age:.2f}, df = {df_diff_age}, p = {p_age_social:.4g}"
    )
    print(
        f"Coefficient p-value for age (social_use): "
        f"{results['social']['p_age_coef']:.4g}"
    )

    social_effects = estimate_effects(
        df, results["social"]["model"], outcome_label="use of social information"
    )

    print("\n=== Preference for majority option ===")
    lr_stat_m, df_diff_m, p_site_majority = results["majority"]["lr_site"]
    print(
        f"LR test for site effect (majority_choice): "
        f"chi2 = {lr_stat_m:.2f}, df = {df_diff_m}, p = {p_site_majority:.4g}"
    )
    lr_stat_age_m, df_diff_age_m, p_age_majority = results["majority"]["lr_age"]
    print(
        f"LR test for age effect (majority_choice): "
        f"chi2 = {lr_stat_age_m:.2f}, df = {df_diff_age_m}, p = {p_age_majority:.4g}"
    )
    print(
        f"Coefficient p-value for age (majority_choice): "
        f"{results['majority']['p_age_coef']:.4g}"
    )

    majority_effects = estimate_effects(
        df, results["majority"]["model"], outcome_label="choice of majority option"
    )

    # Optionally, save a small summary JSON file for manual inspection (not required)
    summary = {
        "social_lr_site_p": float(p_site_social),
        "social_lr_age_p": float(p_age_social),
        "social_age_coef_p": results["social"]["p_age_coef"],
        "majority_lr_site_p": float(p_site_majority),
        "majority_lr_age_p": float(p_age_majority),
        "majority_age_coef_p": results["majority"]["p_age_coef"],
        "social_effects": social_effects,
        "majority_effects": majority_effects,
    }
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


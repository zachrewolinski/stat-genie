import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def load_data():
    base = Path(__file__).parent
    info_path = base / "info.json"
    data_path = base / "boxes.csv"

    with info_path.open("r") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)
    return info, df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Outcome coding:
    # 1 = undemonstrated option, 2 = majority option, 3 = minority option
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    df["social_choice"] = df["majority_first"].isin([2, 3]).astype(int)

    # Site identifier as categorical
    df["site"] = df["y"].astype("category")

    # Age groups for descriptive summaries
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=True,
    )

    return df


def fit_logit(formula: str, df: pd.DataFrame):
    try:
        model = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
        return model
    except Exception:
        # Fallback to GLM Binomial if Logit has convergence issues
        model = smf.glm(formula, data=df, family=smf.families.Binomial()).fit()
        return model


def lr_test(full_model, reduced_model):
    ll_full = full_model.llf
    ll_reduced = reduced_model.llf
    df_full = len(full_model.params)
    df_reduced = len(reduced_model.params)
    df_diff = df_full - df_reduced
    lr_stat = 2 * (ll_full - ll_reduced)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def run_analysis():
    info, df_raw = load_data()
    df = prepare_data(df_raw)

    print("Research question:")
    for q in info.get("research_questions", []):
        print(f" - {q}")
    print()

    n = len(df)
    print(f"Number of observations: {n}")

    # Descriptive summaries by age group and site
    age_group_summary = (
        df.groupby("age_group")[["majority_choice", "social_choice"]]
        .mean()
        .sort_index()
    )
    site_summary = (
        df.groupby("site")[["majority_choice", "social_choice"]]
        .mean()
        .sort_index()
    )

    print("\nProportion of majority and social choices by age group:")
    print(age_group_summary)

    print("\nProportion of majority and social choices by site:")
    print(site_summary)

    # Logistic models for majority_choice
    print("\n=== Logistic models: majority_choice (majority vs other) ===")
    formula_full_maj = "majority_choice ~ age + I(age**2) + C(site) + culture"
    formula_no_site_maj = "majority_choice ~ age + I(age**2) + culture"
    formula_no_age_maj = "majority_choice ~ C(site) + culture"

    maj_full = fit_logit(formula_full_maj, df)
    maj_no_site = fit_logit(formula_no_site_maj, df)
    maj_no_age = fit_logit(formula_no_age_maj, df)

    lr_site_maj, df_site_maj, p_site_maj = lr_test(maj_full, maj_no_site)
    lr_age_maj, df_age_maj, p_age_maj = lr_test(maj_full, maj_no_age)

    print("\nMajority choice model (full) coefficients:")
    print(maj_full.params)
    print("\nP-values (full model):")
    print(maj_full.pvalues)

    print(
        f"\nLR test for site effects on majority_choice: "
        f"LR={lr_site_maj:.3f}, df={df_site_maj}, p={p_site_maj:.4g}"
    )
    print(
        f"LR test for age effects on majority_choice: "
        f"LR={lr_age_maj:.3f}, df={df_age_maj}, p={p_age_maj:.4g}"
    )

    # Logistic models for social_choice
    print("\n=== Logistic models: social_choice (any social vs undemonstrated) ===")
    formula_full_soc = "social_choice ~ age + I(age**2) + C(site) + culture"
    formula_no_site_soc = "social_choice ~ age + I(age**2) + culture"
    formula_no_age_soc = "social_choice ~ C(site) + culture"

    soc_full = fit_logit(formula_full_soc, df)
    soc_no_site = fit_logit(formula_no_site_soc, df)
    soc_no_age = fit_logit(formula_no_age_soc, df)

    lr_site_soc, df_site_soc, p_site_soc = lr_test(soc_full, soc_no_site)
    lr_age_soc, df_age_soc, p_age_soc = lr_test(soc_full, soc_no_age)

    print("\nSocial choice model (full) coefficients:")
    print(soc_full.params)
    print("\nP-values (full model):")
    print(soc_full.pvalues)

    print(
        f"\nLR test for site effects on social_choice: "
        f"LR={lr_site_soc:.3f}, df={df_site_soc}, p={p_site_soc:.4g}"
    )
    print(
        f"LR test for age effects on social_choice: "
        f"LR={lr_age_soc:.3f}, df={df_age_soc}, p={p_age_soc:.4g}"
    )

    # Simple range summaries for later interpretation
    maj_by_age = age_group_summary["majority_choice"]
    soc_by_age = age_group_summary["social_choice"]
    maj_by_site = site_summary["majority_choice"]
    soc_by_site = site_summary["social_choice"]

    print("\nSummary ranges (for interpretation):")
    print(
        "Majority choice by age group: "
        f"min={maj_by_age.min():.3f}, max={maj_by_age.max():.3f}"
    )
    print(
        "Social choice by age group: "
        f"min={soc_by_age.min():.3f}, max={soc_by_age.max():.3f}"
    )
    print(
        "Majority choice by site: "
        f"min={maj_by_site.min():.3f}, max={maj_by_site.max():.3f}"
    )
    print(
        "Social choice by site: "
        f"min={soc_by_site.min():.3f}, max={soc_by_site.max():.3f}"
    )


if __name__ == "__main__":
    run_analysis()


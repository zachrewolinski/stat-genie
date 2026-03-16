import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def summarize_categorical_effects(prefix: str, model, term_prefix: str) -> None:
    """Print a compact summary of categorical (dummy) term p-values."""
    pvals = model.pvalues.filter(like=term_prefix)
    pvals = pvals.dropna()
    if pvals.empty:
        print(f"{prefix}: no terms matching '{term_prefix}'")
        return
    sig_mask = pvals < 0.05
    print(
        f"{prefix}: n_terms={len(pvals)}, "
        f"n_sig(p<0.05)={int(sig_mask.sum())}, "
        f"min_p={pvals.min():.4g}, max_p={pvals.max():.4g}"
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = pd.read_csv("boxes.csv")

    # Basic cleaning: keep rows with complete key variables
    df = df.dropna(
        subset=["y", "gender", "age", "majority_first", "culture"],
    )

    # Encode outcome variants
    df["social"] = df["y"].isin([2, 3]).astype(int)
    social_df = df.copy()

    print("Number of observations:", len(social_df))
    print("Overall social-information use rate:", social_df["social"].mean())

    # Model 1: reliance on social information (any demonstrated option vs undemonstrated)
    print("\n=== Logistic model: social-information use ===")
    social_formula = "social ~ age + C(culture) + gender + majority_first"
    social_model = smf.logit(social_formula, data=social_df).fit(disp=False)

    if "age" in social_model.params:
        age_coef = social_model.params["age"]
        age_p = social_model.pvalues["age"]
        print(f"Age effect on social use: coef={age_coef:.4f}, p={age_p:.4g}")
    summarize_categorical_effects(
        "Culture effects on social use", social_model, "C(culture)"
    )

    # Model 2: majority preference among those who use social information
    social_only = df[df["social"] == 1].copy()
    social_only["majority_choice"] = (social_only["y"] == 2).astype(int)

    print("\n=== Logistic model: majority vs minority choice (conditional on social use) ===")
    maj_formula = "majority_choice ~ age + C(culture) + gender + majority_first"
    maj_model = smf.logit(maj_formula, data=social_only).fit(disp=False)

    if "age" in maj_model.params:
        age_coef_maj = maj_model.params["age"]
        age_p_maj = maj_model.pvalues["age"]
        print(
            f"Age effect on majority preference: coef={age_coef_maj:.4f}, "
            f"p={age_p_maj:.4g}"
        )
    summarize_categorical_effects(
        "Culture effects on majority preference", maj_model, "C(culture)"
    )

    # Simple descriptive summaries by culture and age quartiles to help interpretation.
    social_df["age_group"] = pd.qcut(
        social_df["age"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    social_rate_by_culture = (
        social_df.groupby("culture")["social"].mean().sort_index()
    )
    print("\nSocial-information use rate by culture:")
    print(social_rate_by_culture.to_string())

    social_rate_by_age_group = (
        social_df.groupby("age_group")["social"].mean().sort_index()
    )
    print("\nSocial-information use rate by age quartile:")
    print(social_rate_by_age_group.to_string())

    maj_rate_by_culture = (
        social_only.groupby("culture")["majority_choice"].mean().sort_index()
    )
    print("\nMajority-choice rate by culture (among social users):")
    print(maj_rate_by_culture.to_string())

    social_only["age_group"] = pd.qcut(
        social_only["age"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    maj_rate_by_age_group = (
        social_only.groupby("age_group")["majority_choice"].mean().sort_index()
    )
    print("\nMajority-choice rate by age quartile (among social users):")
    print(maj_rate_by_age_group.to_string())


if __name__ == "__main__":
    main()


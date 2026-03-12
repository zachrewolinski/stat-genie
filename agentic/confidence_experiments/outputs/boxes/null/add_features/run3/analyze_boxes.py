import json
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def lr_test(model_restricted, model_full) -> Tuple[float, int, float]:
    """Likelihood-ratio test comparing two nested statsmodels GLMs."""
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    df_diff = int(round(model_full.df_model - model_restricted.df_model))
    p_value = float(stats.chi2.sf(lr_stat, df_diff))
    return lr_stat, df_diff, p_value


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Define key derived variables
    df["social"] = (df["y"] != 1).astype(int)
    social_df = df[df["social"] == 1].copy()
    social_df["majority_choice"] = (social_df["y"] == 2).astype(int)

    # Age groups following the coding described in metadata
    bins = [17.5, 25, 35, 45, 60]
    labels = ["18-24", "25-34", "35-44", "45-57"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)
    social_df["age_group"] = pd.cut(
        social_df["age"], bins=bins, labels=labels, right=False, include_lowest=True
    )

    print("=== Overall social information use ===")
    social_rate = df["social"].mean()
    print(f"Proportion using social information (y != 1): {social_rate:.3f}")

    print("\n=== Overall majority preference among social learners ===")
    majority_rate = social_df["majority_choice"].mean()
    print(f"Proportion choosing majority option among social learners: {majority_rate:.3f}")

    # --- Chi-square tests: Social use vs age group and culture ---
    print("\n=== Social information use by age group (chi-square) ===")
    age_social_table = pd.crosstab(df["age_group"], df["social"])
    print(age_social_table)
    chi2_age_social, p_age_social, dof_age_social, _ = stats.chi2_contingency(age_social_table)
    print(f"Chi2={chi2_age_social:.3f}, df={dof_age_social}, p={p_age_social:.4f}")

    print("\n=== Social information use by culture (chi-square) ===")
    culture_social_table = pd.crosstab(df["culture"], df["social"])
    print(culture_social_table)
    chi2_cult_social, p_cult_social, dof_cult_social, _ = stats.chi2_contingency(culture_social_table)
    print(f"Chi2={chi2_cult_social:.3f}, df={dof_cult_social}, p={p_cult_social:.4f}")

    # --- Chi-square tests: Majority vs minority choice among social learners ---
    print("\n=== Majority vs minority choice by age group (chi-square; social learners only) ===")
    age_majority_table = pd.crosstab(social_df["age_group"], social_df["majority_choice"])
    print(age_majority_table)
    chi2_age_majority, p_age_majority, dof_age_majority, _ = stats.chi2_contingency(age_majority_table)
    print(f"Chi2={chi2_age_majority:.3f}, df={dof_age_majority}, p={p_age_majority:.4f}")

    print("\n=== Majority vs minority choice by culture (chi-square; social learners only) ===")
    culture_majority_table = pd.crosstab(social_df["culture"], social_df["majority_choice"])
    print(culture_majority_table)
    chi2_cult_majority, p_cult_majority, dof_cult_majority, _ = stats.chi2_contingency(culture_majority_table)
    print(f"Chi2={chi2_cult_majority:.3f}, df={dof_cult_majority}, p={p_cult_majority:.4f}")

    # --- Logistic regression: Social use ---
    print("\n=== Logistic regression: Social information use ===")
    # Baseline model without age or culture
    m0_social = smf.logit("social ~ C(gender) + majority_first", data=df).fit(disp=False, maxiter=200)
    # Add age
    m1_social = smf.logit("social ~ age + C(gender) + majority_first", data=df).fit(disp=False, maxiter=200)
    lr_age_social, df_age_social, p_lr_age_social = lr_test(m0_social, m1_social)
    print("Age effect (LR test, adding age):")
    print(f"  LR stat={lr_age_social:.3f}, df={df_age_social}, p={p_lr_age_social:.4f}")
    print(m1_social.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]])

    # Add culture
    m2_social = smf.logit(
        "social ~ age + C(culture) + C(gender) + majority_first", data=df
    ).fit(disp=False, maxiter=200)
    lr_cult_social, df_cult_social, p_lr_cult_social = lr_test(m1_social, m2_social)
    print("\nCulture effect (LR test, adding C(culture)):")
    print(f"  LR stat={lr_cult_social:.3f}, df={df_cult_social}, p={p_lr_cult_social:.4f}")

    # --- Logistic regression: Majority vs minority among social learners ---
    print("\n=== Logistic regression: Majority vs minority choice (social learners only) ===")
    m0_majority = smf.logit(
        "majority_choice ~ C(gender) + majority_first", data=social_df
    ).fit(disp=False, maxiter=200)
    m1_majority = smf.logit(
        "majority_choice ~ age + C(gender) + majority_first", data=social_df
    ).fit(disp=False, maxiter=200)
    lr_age_majority, df_age_majority_lr, p_lr_age_majority = lr_test(m0_majority, m1_majority)
    print("Age effect (LR test, adding age):")
    print(f"  LR stat={lr_age_majority:.3f}, df={df_age_majority_lr}, p={p_lr_age_majority:.4f}")
    print(m1_majority.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]])

    m2_majority = smf.logit(
        "majority_choice ~ age + C(culture) + C(gender) + majority_first", data=social_df
    ).fit(disp=False, maxiter=200)
    lr_cult_majority, df_cult_majority_lr, p_lr_cult_majority = lr_test(m1_majority, m2_majority)
    print("\nCulture effect (LR test, adding C(culture)):")
    print(f"  LR stat={lr_cult_majority:.3f}, df={df_cult_majority_lr}, p={p_lr_cult_majority:.4f}")

    # The script itself does not write conclusion.txt; that is produced separately
    # after interpreting these results.


if __name__ == "__main__":
    main()


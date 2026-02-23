import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Compute Cramer's V effect size."""
    if n == 0:
        return np.nan
    k = min(r - 1, c - 1)
    if k == 0:
        return np.nan
    return np.sqrt(chi2 / (n * k))


def chi_square_with_effect(table: pd.DataFrame):
    chi2, p, dof, expected = chi2_contingency(table)
    v = cramers_v(chi2, table.to_numpy().sum(), *table.shape)
    return chi2, p, dof, v


def main():
    df = pd.read_csv("boxes.csv")

    # Define key outcomes
    df["social"] = (df["y"] != 1).astype(int)  # 1 if majority or minority option chosen
    df["majority"] = (df["y"] == 2).astype(int)  # 1 if majority option chosen

    # Age groups approximating developmental stages in this dataset
    # The age variable is coded in coarse bands (e.g., 17.5, 22, 27, 32, ...).
    # We bin these into three ordered stages for analysis.
    bins = [0, 25, 35, 100]
    labels = ["<25", "25-34", "35+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    print("Sample size:", len(df))
    print("\nOverall outcome distribution (y):")
    print(df["y"].value_counts(normalize=True).sort_index())

    # Proportions of social learning and majority choice by culture
    print("\nProportion using social information (by culture):")
    social_by_culture = df.groupby("culture")["social"].mean()
    print(social_by_culture)

    print("\nProportion choosing majority option (by culture):")
    majority_by_culture = df.groupby("culture")["majority"].mean()
    print(majority_by_culture)

    # Proportions by age group
    print("\nProportion using social information (by age_group):")
    social_by_age = df.groupby("age_group")["social"].mean()
    print(social_by_age)

    print("\nProportion choosing majority option (by age_group):")
    majority_by_age = df.groupby("age_group")["majority"].mean()
    print(majority_by_age)

    # Chi-square tests for associations
    print("\nChi-square tests with Cramer's V:")

    # social vs culture
    tab_sc = pd.crosstab(df["social"], df["culture"])
    chi2, p, dof, v = chi_square_with_effect(tab_sc)
    print(f"social ~ culture: chi2={chi2:.3f}, dof={dof}, p={p:.4g}, V={v:.3f}")

    # majority vs culture
    tab_mc = pd.crosstab(df["majority"], df["culture"])
    chi2, p, dof, v = chi_square_with_effect(tab_mc)
    print(f"majority ~ culture: chi2={chi2:.3f}, dof={dof}, p={p:.4g}, V={v:.3f}")

    # social vs age_group
    tab_sa = pd.crosstab(df["social"], df["age_group"])
    chi2, p, dof, v = chi_square_with_effect(tab_sa)
    print(f"social ~ age_group: chi2={chi2:.3f}, dof={dof}, p={p:.4g}, V={v:.3f}")

    # majority vs age_group
    tab_ma = pd.crosstab(df["majority"], df["age_group"])
    chi2, p, dof, v = chi_square_with_effect(tab_ma)
    print(f"majority ~ age_group: chi2={chi2:.3f}, dof={dof}, p={p:.4g}, V={v:.3f}")


if __name__ == "__main__":
    main()


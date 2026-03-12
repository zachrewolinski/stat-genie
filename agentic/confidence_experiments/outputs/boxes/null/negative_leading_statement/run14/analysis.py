import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["social_choice"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Define age groups for developmental stages
    bins = [4, 6, 9, 12, 14]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True, right=True)

    print("Sample size:", len(df))
    print()

    # Overall distribution of choices
    print("Outcome distribution (y: 1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts().sort_index())
    print()

    # Chi-square: social vs non-social by culture
    print("=== Social information use by culture ===")
    social_culture = pd.crosstab(df["culture"], df["social_choice"])
    chi2, p, dof, expected = chi2_contingency(social_culture)
    print("Contingency table (rows=culture, cols=social_choice 0/1):")
    print(social_culture)
    print(f"chi2={chi2:.3f}, dof={dof}, p={p:.5f}")
    print()

    # Chi-square: majority choice by culture
    print("=== Majority choice vs other responses by culture ===")
    majority_culture = pd.crosstab(df["culture"], df["majority_choice"])
    chi2_m, p_m, dof_m, expected_m = chi2_contingency(majority_culture)
    print("Contingency table (rows=culture, cols=majority_choice 0/1):")
    print(majority_culture)
    print(f"chi2={chi2_m:.3f}, dof={dof_m}, p={p_m:.5f}")
    print()

    # Chi-square: social vs non-social by age group
    print("=== Social information use by age group ===")
    social_age = pd.crosstab(df["age_group"], df["social_choice"])
    chi2_a, p_a, dof_a, expected_a = chi2_contingency(social_age)
    print("Contingency table (rows=age_group, cols=social_choice 0/1):")
    print(social_age)
    print(f"chi2={chi2_a:.3f}, dof={dof_a}, p={p_a:.5f}")
    print()

    # Chi-square: majority choice by age group
    print("=== Majority choice vs other responses by age group ===")
    majority_age = pd.crosstab(df["age_group"], df["majority_choice"])
    chi2_ma, p_ma, dof_ma, expected_ma = chi2_contingency(majority_age)
    print("Contingency table (rows=age_group, cols=majority_choice 0/1):")
    print(majority_age)
    print(f"chi2={chi2_ma:.3f}, dof={dof_ma}, p={p_ma:.5f}")
    print()


if __name__ == "__main__":
    main()


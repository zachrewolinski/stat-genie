import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic recoding
    df["is_majority"] = (df["y"] == 2).astype(int)
    df["is_minority"] = (df["y"] == 3).astype(int)
    df["is_undemonstrated"] = (df["y"] == 1).astype(int)
    df["uses_social_info"] = (df["y"] != 1).astype(int)

    # Create coarse age groups to study developmental stages
    age_bins = [4, 6, 9, 12, 14]
    age_labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, include_lowest=True, right=True)

    print("N observations:", len(df))
    print()

    # Overall outcome distribution
    print("Overall choice distribution (1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts(normalize=False).sort_index())
    print("Proportions:")
    print(df["y"].value_counts(normalize=True).sort_index())
    print()

    # Majority choice by culture
    print("Proportion choosing majority option by culture:")
    majority_by_culture = df.groupby("culture")["is_majority"].mean()
    print(majority_by_culture)
    print("Range across cultures:", majority_by_culture.min(), "to", majority_by_culture.max())
    print()

    # Majority choice by age group
    print("Proportion choosing majority option by age group:")
    majority_by_age = df.groupby("age_group")["is_majority"].mean()
    print(majority_by_age)
    print("Range across age groups:", majority_by_age.min(), "to", majority_by_age.max())
    print()

    # Reliance on any social information (majority or minority) by culture and age
    print("Proportion relying on social info (not undemonstrated) by culture:")
    social_by_culture = df.groupby("culture")["uses_social_info"].mean()
    print(social_by_culture)
    print("Range across cultures:", social_by_culture.min(), "to", social_by_culture.max())
    print()

    print("Proportion relying on social info by age group:")
    social_by_age = df.groupby("age_group")["uses_social_info"].mean()
    print(social_by_age)
    print("Range across age groups:", social_by_age.min(), "to", social_by_age.max())
    print()

    # Chi-squared tests for variation across culture and age_group
    print("Chi-squared tests:")
    culture_y_table = pd.crosstab(df["culture"], df["y"])
    chi2_cult, p_cult, dof_cult, _ = chi2_contingency(culture_y_table)
    print("Culture x outcome: chi2 =", chi2_cult, "df =", dof_cult, "p =", p_cult)

    age_y_table = pd.crosstab(df["age_group"], df["y"])
    chi2_age, p_age, dof_age, _ = chi2_contingency(age_y_table)
    print("Age group x outcome: chi2 =", chi2_age, "df =", dof_age, "p =", p_age)
    print()

    # Simple check: majority preference increases with age?
    print("Mean majority choice by exact age:")
    print(df.groupby("age")["is_majority"].mean())


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["follow_social"] = df["y"].isin([2, 3]).astype(int)
    df["age_c"] = df["age"] - df["age"].mean()

    social_df = df[df["y"].isin([2, 3])].copy()
    social_df["choose_majority"] = (social_df["y"] == 2).astype(int)
    social_df["age_c"] = social_df["age"] - social_df["age"].mean()

    # Age groups to approximate developmental stages
    age_bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    age_labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
    social_df["age_group"] = pd.cut(social_df["age"], bins=age_bins, labels=age_labels)

    print("Basic sample description")
    print("------------------------")
    print("N =", len(df))
    print("\nOutcome distribution (y: 1=undemonstrated, 2=majority, 3=minority)")
    print(df["y"].value_counts().sort_index())

    print("\nReliance on social information (follow_social = 1 if majority/minority, 0 if undemonstrated)")
    print("Overall mean follow_social:", df["follow_social"].mean())
    print("\nBy culture:")
    print(df.groupby("culture")["follow_social"].mean())
    print("\nBy age_group:")
    print(df.groupby("age_group")["follow_social"].mean())

    print("\nPreference for majority among social choices (choose_majority)")
    print("Overall mean choose_majority:", social_df["choose_majority"].mean())
    print("\nBy culture:")
    print(social_df.groupby("culture")["choose_majority"].mean())
    print("\nBy age_group:")
    print(social_df.groupby("age_group")["choose_majority"].mean())

    # Logistic regression: reliance on social information
    print("\nLogistic regression: follow_social ~ age_c + C(culture) + gender + majority_first")
    model_follow = smf.logit(
        "follow_social ~ age_c + C(culture) + gender + majority_first",
        data=df,
    ).fit(disp=False)
    print(model_follow.summary())

    # Logistic regression: majority preference among social choices
    print("\nLogistic regression: choose_majority ~ age_c + C(culture) + gender + majority_first")
    model_majority = smf.logit(
        "choose_majority ~ age_c + C(culture) + gender + majority_first",
        data=social_df,
    ).fit(disp=False)
    print(model_majority.summary())

    # Chi-square tests for associations
    print("\nChi-square tests of association")

    # Reliance on social information by culture
    ct_follow_culture = pd.crosstab(df["culture"], df["follow_social"])
    chi2_fc, p_fc, dof_fc, _ = chi2_contingency(ct_follow_culture)
    print("\nFollow_social by culture:")
    print(ct_follow_culture)
    print(f"chi2 = {chi2_fc:.3f}, dof = {dof_fc}, p = {p_fc:.4f}")

    # Reliance on social information by age group
    ct_follow_age = pd.crosstab(df["age_group"], df["follow_social"])
    chi2_fa, p_fa, dof_fa, _ = chi2_contingency(ct_follow_age)
    print("\nFollow_social by age_group:")
    print(ct_follow_age)
    print(f"chi2 = {chi2_fa:.3f}, dof = {dof_fa}, p = {p_fa:.4f}")

    # Majority preference by culture
    ct_maj_culture = pd.crosstab(social_df["culture"], social_df["choose_majority"])
    chi2_mc, p_mc, dof_mc, _ = chi2_contingency(ct_maj_culture)
    print("\nChoose_majority by culture:")
    print(ct_maj_culture)
    print(f"chi2 = {chi2_mc:.3f}, dof = {dof_mc}, p = {p_mc:.4f}")

    # Majority preference by age group
    ct_maj_age = pd.crosstab(social_df["age_group"], social_df["choose_majority"])
    chi2_ma, p_ma, dof_ma, _ = chi2_contingency(ct_maj_age)
    print("\nChoose_majority by age_group:")
    print(ct_maj_age)
    print(f"chi2 = {chi2_ma:.3f}, dof = {dof_ma}, p = {p_ma:.4f}")


if __name__ == "__main__":
    main()

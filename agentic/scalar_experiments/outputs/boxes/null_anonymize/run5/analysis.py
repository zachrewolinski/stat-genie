import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # feature1: 1 = undemonstrated option, 2 = majority option, 3 = minority option
    df["social_reliance"] = df["feature1"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["feature1"] == 2).astype(int)

    # Overall behavior
    overall_counts = df["feature1"].value_counts().sort_index()
    n = len(df)

    print("N:", n)
    print("Outcome counts (1=undemonstrated,2=majority,3=minority):")
    print(overall_counts)
    print("Proportions:")
    print((overall_counts / n).round(3))

    # Age effects on reliance on social information (any demonstrated choice)
    X_age = sm.add_constant(df["feature3"])
    model_social = sm.Logit(df["social_reliance"], X_age).fit(disp=False)
    print("\nLogit: social_reliance ~ age")
    print(model_social.summary())

    # Age effects on majority vs other responses among those who used social info
    df_social = df[df["social_reliance"] == 1].copy()
    X_age_pref = sm.add_constant(df_social["feature3"])
    model_majority = sm.Logit(df_social["majority_choice"], X_age_pref).fit(disp=False)
    print("\nLogit: majority_choice ~ age (among social learners)")
    print(model_majority.summary())

    # Site (culture) effects on majority choice
    contingency = pd.crosstab(df["feature5"], df["majority_choice"])
    chi2, p_site, dof, expected = stats.chi2_contingency(contingency)
    print("\nSite x majority_choice contingency:")
    print(contingency)
    print(f"Chi2 for site effect on majority_choice: {chi2:.3f}, p={p_site:.3g}, dof={dof}")

    # Site effects on overall social reliance
    contingency_social = pd.crosstab(df["feature5"], df["social_reliance"])
    chi2_sr, p_site_sr, dof_sr, _ = stats.chi2_contingency(contingency_social)
    print("\nSite x social_reliance contingency:")
    print(contingency_social)
    print(
        f"Chi2 for site effect on social_reliance: {chi2_sr:.3f}, "
        f"p={p_site_sr:.3g}, dof={dof_sr}"
    )

    # Simple descriptive variation measures
    majority_by_site = df.groupby("feature5")["majority_choice"].mean()
    social_by_site = df.groupby("feature5")["social_reliance"].mean()
    print("\nMean majority_choice by site:")
    print(majority_by_site.round(3))
    print("\nMean social_reliance by site:")
    print(social_by_site.round(3))

    # Age-binned descriptions
    df["age_bin"] = pd.cut(df["feature3"], bins=[4, 6, 8, 10, 12, 14], include_lowest=True)
    majority_by_age = df.groupby("age_bin")["majority_choice"].mean()
    social_by_age = df.groupby("age_bin")["social_reliance"].mean()
    print("\nMean majority_choice by age_bin:")
    print(majority_by_age.round(3))
    print("\nMean social_reliance by age_bin:")
    print(social_by_age.round(3))


if __name__ == "__main__":
    main()


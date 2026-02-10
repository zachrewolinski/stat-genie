import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Outcomes
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Among social choices, is the child following the majority?
    social_df = df[df["social_choice"] == 1].copy()

    # Simple age bands to capture developmental stages
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True, right=True)
    social_df["age_group"] = pd.cut(
        social_df["age"], bins=bins, labels=labels, include_lowest=True, right=True
    )

    print("=== Sample size ===")
    print(len(df))
    print()

    print("=== Overall choice distribution (1=undemonstrated, 2=majority, 3=minority) ===")
    print(df["y"].value_counts(normalize=True).sort_index())
    print()

    print("=== Social information use by age group ===")
    social_by_age = (
        df.groupby("age_group")["social_choice"]
        .mean()
        .rename("p_social")
        .to_frame()
    )
    print(social_by_age)
    print()

    print("=== Social information use by culture ===")
    social_by_culture = (
        df.groupby("culture")["social_choice"]
        .mean()
        .rename("p_social")
        .to_frame()
    )
    print(social_by_culture)
    print()

    print("=== Majority preference (among social choices) by age group ===")
    maj_by_age = (
        social_df.groupby("age_group")["majority_choice"]
        .mean()
        .rename("p_majority_given_social")
        .to_frame()
    )
    print(maj_by_age)
    print()

    print("=== Majority preference (among social choices) by culture ===")
    maj_by_culture = (
        social_df.groupby("culture")["majority_choice"]
        .mean()
        .rename("p_majority_given_social")
        .to_frame()
    )
    print(maj_by_culture)
    print()

    # Logistic regression: reliance on social information ~ age + culture
    print("=== GLM: social_choice ~ age + culture ===")
    model_social = smf.glm(
        "social_choice ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_social.summary())
    print()

    # Logistic regression: majority preference among social choices ~ age + culture
    print("=== GLM: majority_choice (social only) ~ age + culture ===")
    model_majority = smf.glm(
        "majority_choice ~ age + C(culture)",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_majority.summary())


if __name__ == "__main__":
    main()


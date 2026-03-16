import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Social-information reliance: choosing any demonstrated option (majority or minority)
    df["social_choice"] = (df["y"] != 1).astype(int)

    # Majority preference among social choices only
    social_df = df[df["y"] != 1].copy()
    social_df["majority_choice"] = (social_df["y"] == 2).astype(int)

    print("Dataset shape:", df.shape)
    print(df.head(), "\n")

    # Logistic/GLM models
    print("=== Model 1: Social-information reliance (any demonstrated option vs undemonstrated) ===")
    model_social = smf.glm(
        "social_choice ~ age + C(culture) + gender + majority_first",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_social.summary())
    print("\n")

    print("=== Model 2: Majority preference among social choices (majority vs minority) ===")
    model_majority = smf.glm(
        "majority_choice ~ age + C(culture) + gender + majority_first",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_majority.summary())
    print("\n")

    # Simple descriptive statistics by age and culture
    print("=== Descriptive: mean social_choice by age ===")
    print(df.groupby("age")["social_choice"].mean())
    print("\n")

    print("=== Descriptive: mean majority_choice (among social choices) by age ===")
    print(social_df.groupby("age")["majority_choice"].mean())
    print("\n")

    print("=== Descriptive: mean social_choice by culture ===")
    print(df.groupby("culture")["social_choice"].mean())
    print("\n")

    print("=== Descriptive: mean majority_choice (among social choices) by culture ===")
    print(social_df.groupby("culture")["majority_choice"].mean())
    print("\n")

    # Categorical age to check for non-linear developmental patterns
    print("=== Model 1b: Social-information reliance with categorical age ===")
    model_social_cat_age = smf.glm(
        "social_choice ~ C(age) + C(culture) + gender + majority_first",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_social_cat_age.summary())
    print("\n")

    print("=== Model 2b: Majority preference with categorical age ===")
    model_majority_cat_age = smf.glm(
        "majority_choice ~ C(age) + C(culture) + gender + majority_first",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_majority_cat_age.summary())
    print("\n")


if __name__ == "__main__":
    main()

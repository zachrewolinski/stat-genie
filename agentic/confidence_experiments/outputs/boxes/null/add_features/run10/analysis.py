import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Keep only columns we need
    cols = ["y", "gender", "age", "majority_first", "culture"]
    df = df[cols].copy()

    # Drop rows with missing values in relevant columns
    df = df.dropna(subset=["y", "age", "culture"])

    # Ensure integer coding for outcome
    df["y"] = df["y"].astype(int)

    # Derived variables
    # Reliance on social information: choosing any demonstrated option (majority or minority)
    df["social_use"] = (df["y"] != 1).astype(int)

    # Majority preference among those using social information
    df["majority_choice"] = np.where(
        df["y"] == 2,
        1,
        np.where(df["y"] == 3, 0, np.nan),
    )

    # Fit logistic-type models via Binomial GLM to avoid separation errors
    social_model = smf.glm(
        "social_use ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    df_social = df[df["social_use"] == 1].copy()
    majority_model = smf.glm(
        "majority_choice ~ age + C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    # Basic descriptive statistics
    print("N total:", len(df))
    print("Social-use proportion overall:", df["social_use"].mean())
    print(
        "Majority preference among social users:",
        df_social["majority_choice"].mean(),
    )

    print("\nSocial-use model summary:")
    print(social_model.summary())

    print("\nMajority-choice model summary:")
    print(majority_model.summary())

    print("\nSocial-use by culture:")
    print(df.groupby("culture")["social_use"].mean())

    print("\nMajority-choice among social users by culture:")
    print(df_social.groupby("culture")["majority_choice"].mean())

    print("\nSocial-use by age tertile:")
    df["age_tertile"] = pd.qcut(
        df["age"],
        3,
        labels=["young", "mid", "old"],
        duplicates="drop",
    )
    print(df.groupby("age_tertile")["social_use"].mean())

    print("\nMajority-choice by age tertile:")
    df_social["age_tertile"] = pd.qcut(
        df_social["age"],
        3,
        labels=["young", "mid", "old"],
        duplicates="drop",
    )
    print(df_social.groupby("age_tertile")["majority_choice"].mean())


if __name__ == "__main__":
    main()


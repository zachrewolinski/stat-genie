import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Social information use: choosing either majority or minority demonstrator
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)

    # Majority preference among children who follow some demonstrator
    df["majority_choice_among_social"] = np.where(
        df["y"] == 2,
        1,
        np.where(df["y"] == 3, 0, np.nan),
    )

    print("N total:", len(df))
    print("Overall social-choice rate:", df["social_choice"].mean())
    print(
        "Overall majority-among-social rate:",
        df["majority_choice_among_social"].mean(),
    )

    # Model 1: reliance on social information
    print("\n=== Logistic model: social_choice ~ age + culture + gender + majority_first ===")
    model_social = smf.logit(
        "social_choice ~ age + C(culture) + C(gender) + majority_first", data=df
    ).fit(disp=False)
    print(model_social.summary())

    # Descriptives by age and culture for social_choice
    print("\nSocial-choice rate by age:")
    print(df.groupby("age")["social_choice"].mean())

    print("\nSocial-choice rate by culture:")
    print(df.groupby("culture")["social_choice"].mean())

    # Model 2: majority preference among those who use social information
    df_social = df[df["social_choice"] == 1].copy()
    print("\nN with social_choice=1:", len(df_social))

    print(
        "\n=== Logistic model: majority_choice_among_social ~ age + culture + gender + majority_first ==="
    )
    model_majority = smf.logit(
        "majority_choice_among_social ~ age + C(culture) + C(gender) + majority_first",
        data=df_social,
    ).fit(disp=False)
    print(model_majority.summary())

    # Descriptives by age and culture for majority_choice_among_social
    print("\nMajority-among-social rate by age:")
    print(df_social.groupby("age")["majority_choice_among_social"].mean())

    print("\nMajority-among-social rate by culture:")
    print(df_social.groupby("culture")["majority_choice_among_social"].mean())


if __name__ == "__main__":
    main()


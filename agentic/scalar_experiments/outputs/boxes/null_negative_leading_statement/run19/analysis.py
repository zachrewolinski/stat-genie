import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic structure
    print("Shape:", df.shape)
    print("\nOutcome counts (y):")
    print(df["y"].value_counts().sort_index())

    # Recode outcome into majority-choice indicator
    df["majority_choice"] = (df["y"] == 2).astype(int)

    print("\nOverall majority choice rate:")
    print(df["majority_choice"].mean())

    # Reliance on social information: choosing either demonstrated option
    df["social_use"] = (df["y"] != 1).astype(int)
    print("\nOverall social-information use rate (majority or minority):")
    print(df["social_use"].mean())

    # Majority choice by culture
    print("\nMajority choice rate by culture:")
    print(
        df.groupby("culture")["majority_choice"]
        .mean()
        .to_frame("majority_rate")
    )
    print("\nSocial-information use rate by culture:")
    print(
        df.groupby("culture")["social_use"]
        .mean()
        .to_frame("social_use_rate")
    )

    # Define age bands (rough developmental stages)
    age_bins = [4, 6, 8, 10, 12, 14]
    age_labels = ["4-5", "6-7", "8-9", "10-11", "12-13"]
    df["age_band"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, include_lowest=True)

    print("\nMajority choice rate by age band:")
    print(
        df.groupby("age_band")["majority_choice"]
        .mean()
        .to_frame("majority_rate")
    )
    print("\nSocial-information use rate by age band:")
    print(
        df.groupby("age_band")["social_use"]
        .mean()
        .to_frame("social_use_rate")
    )

    # Logistic regression: majority choice as a function of age and culture
    # (culture treated as categorical)
    # Model 1: majority choice as a function of age and culture
    formula1 = "majority_choice ~ age + C(culture)"
    print("\nLogistic regression (Model 1):", formula1)
    logit_model1 = smf.logit(formula=formula1, data=df).fit(disp=False)
    print(logit_model1.summary())

    # Model 2: social-information use as a function of age and culture
    formula2 = "social_use ~ age + C(culture)"
    print("\nLogistic regression (Model 2):", formula2)
    logit_model2 = smf.logit(formula=formula2, data=df).fit(disp=False)
    print(logit_model2.summary())

    # Model 3: among children who used social information,
    # preference for majority over minority as a function of age and culture
    df_social = df[df["social_use"] == 1].copy()
    print("\nAmong social users only (N = {}):".format(len(df_social)))
    print("Majority choice rate:", df_social["majority_choice"].mean())
    formula3 = "majority_choice ~ age + C(culture)"
    print("\nLogistic regression (Model 3, social users only):", formula3)
    logit_model3 = smf.logit(formula=formula3, data=df_social).fit(disp=False)
    print(logit_model3.summary())


if __name__ == "__main__":
    main()

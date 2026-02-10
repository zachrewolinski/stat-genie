import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Basic overview
    print("Head of data:")
    print(df.head())
    print("\nOutcome distribution (y: 1=third option, 2=majority, 3=minority):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nAge summary:")
    print(df["age"].describe())

    print("\nCulture counts:")
    print(df["culture"].value_counts().sort_index())

    # Create variables approximating social reliance and majority preference
    df["social_choice"] = (df["y"] != 1).astype(int)
    social_rate_overall = df["social_choice"].mean()
    print(f"\nOverall proportion choosing any social option: {social_rate_overall:.3f}")

    df["majority_choice"] = (df["y"] == 2).astype(int)
    majority_rate_overall = df["majority_choice"].mean()
    print(f"Overall proportion choosing majority option: {majority_rate_overall:.3f}")

    # Discrete age groups to approximate developmental stages
    df["age_group"] = pd.qcut(df["age"], q=3, labels=["younger", "middle", "older"])
    print("\nSocial choice rate by culture and age_group:")
    social_by_group = (
        df.groupby(["culture", "age_group"])["social_choice"].mean().unstack()
    )
    print(social_by_group)

    print("\nMajority choice rate (among social choosers) by culture and age_group:")
    social_df = df[df["social_choice"] == 1]
    majority_by_group = (
        social_df.groupby(["culture", "age_group"])["majority_choice"].mean().unstack()
    )
    print(majority_by_group)

    # Logistic regression: any social choice vs non-social, as a function of culture and age
    print(
        "\n==== Logistic regression: social_choice ~ age + culture (baseline for variability) ===="
    )
    try:
        model_social = smf.logit("social_choice ~ age + C(culture)", data=df).fit(
            disp=False
        )
        print(model_social.summary())
    except Exception as exc:
        print("social_choice model failed:", exc)

    # Logistic regression: majority vs minority among social choosers
    print(
        "\n==== Logistic regression: majority_choice ~ age + culture (social choosers only) ===="
    )
    try:
        model_majority = smf.logit(
            "majority_choice ~ age + C(culture)", data=social_df
        ).fit(disp=False)
        print(model_majority.summary())
    except Exception as exc:
        print("majority_choice model failed:", exc)


if __name__ == "__main__":
    main()

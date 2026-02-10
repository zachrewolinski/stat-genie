import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Majority choice as primary measure of reliance on social (majority) information
    df["majority_choice"] = (df["y"] == 2).astype(int)

    print("N =", len(df))
    print("Overall majority choice rate:", df["majority_choice"].mean())

    # Descriptive differences by culture
    by_culture = df.groupby("culture")["majority_choice"].agg(["mean", "count"])
    print("\nMajority choice by culture:")
    print(by_culture)
    print("Range of culture means:",
          float(by_culture["mean"].min()),
          "to",
          float(by_culture["mean"].max()))

    # Descriptive differences across age (quartiles)
    df["age_group"] = pd.qcut(df["age"], 4, duplicates="drop")
    by_age_group = df.groupby("age_group")["majority_choice"].agg(["mean", "count"])
    print("\nMajority choice by age quartile:")
    print(by_age_group)

    # Logistic regression: majority_choice ~ age + culture
    full_model = smf.glm(
        "majority_choice ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    print("\nFull model summary (majority_choice ~ age + culture):")
    print(full_model.summary())

    # Likelihood-ratio tests for age and culture effects
    model_no_culture = smf.glm(
        "majority_choice ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    model_no_age = smf.glm(
        "majority_choice ~ C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    def lr_test(restricted, full, df_diff: int, label: str) -> None:
        lr_stat = 2 * (full.llf - restricted.llf)
        p_value = sm.stats.distributions.chisqprob(lr_stat, df_diff)
        print(f"\nLR test for {label}:")
        print("  LR stat:", lr_stat)
        print("  df diff:", df_diff)
        print("  p-value:", p_value)

    # Degrees of freedom differences
    df_diff_culture = full_model.df_model - model_no_culture.df_model
    df_diff_age = full_model.df_model - model_no_age.df_model

    lr_test(model_no_culture, full_model, int(df_diff_culture), "adding culture")
    lr_test(model_no_age, full_model, int(df_diff_age), "adding age")


if __name__ == "__main__":
    main()


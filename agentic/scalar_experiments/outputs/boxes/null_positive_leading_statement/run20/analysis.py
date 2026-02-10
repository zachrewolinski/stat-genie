import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic derived variables
    df["social_reliance"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Age as numeric and simple age bins for descriptive stats
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[4, 6, 8, 10, 12, 14],
        labels=["4-6", "6-8", "8-10", "10-12", "12-14"],
        include_lowest=True,
        right=False,
    )

    print("N observations:", len(df))
    print("\nOverall choice distribution (1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nSocial reliance rate (choose demonstrated option) by culture:")
    rel_by_culture = df.groupby("culture")["social_reliance"].mean()
    print(rel_by_culture)

    print("\nSocial reliance rate by age_bin:")
    rel_by_age = df.groupby("age_bin")["social_reliance"].mean()
    print(rel_by_age)

    # Restrict to demonstrated choices for majority-vs-minority preference
    df_demo = df[df["y"].isin([2, 3])].copy()
    print("\nN demonstrated-only choices:", len(df_demo))

    print("\nMajority choice rate among demonstrated options by culture:")
    maj_by_culture = df_demo.groupby("culture")["majority_choice"].mean()
    print(maj_by_culture)

    print("\nMajority choice rate among demonstrated options by age_bin:")
    maj_by_age = df_demo.groupby("age_bin")["majority_choice"].mean()
    print(maj_by_age)

    # Logistic regression: social reliance ~ age + culture
    df["age_c"] = df["age"] - df["age"].mean()
    df["age_c2"] = df["age_c"] ** 2

    print("\nLogistic regression: social_reliance ~ age + age^2 + culture")
    model_rel = smf.logit("social_reliance ~ age_c + age_c2 + C(culture)", data=df).fit(
        disp=False
    )
    print(model_rel.summary())

    # Logistic regression: majority preference among demonstrated choices
    df_demo["age_c"] = df_demo["age"] - df_demo["age"].mean()
    df_demo["age_c2"] = df_demo["age_c"] ** 2

    print("\nLogistic regression: majority_choice ~ age + age^2 + culture (demonstrated only)")
    model_maj = smf.logit("majority_choice ~ age_c + age_c2 + C(culture)", data=df_demo).fit(
        disp=False
    )
    print(model_maj.summary())

    # Simple measures of variation (range of rates)
    rel_range_culture = rel_by_culture.max() - rel_by_culture.min()
    rel_range_age = rel_by_age.max() - rel_by_age.min()
    maj_range_culture = maj_by_culture.max() - maj_by_culture.min()
    maj_range_age = maj_by_age.max() - maj_by_age.min()

    print("\nRange of social reliance by culture:", rel_range_culture)
    print("Range of social reliance by age_bin:", rel_range_age)
    print("Range of majority preference by culture:", maj_range_culture)
    print("Range of majority preference by age_bin:", maj_range_age)


if __name__ == "__main__":
    main()


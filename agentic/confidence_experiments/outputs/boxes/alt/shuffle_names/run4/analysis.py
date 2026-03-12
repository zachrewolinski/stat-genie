import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Recode variables based on the metadata description.
    # majority_first: 1=unchosen option, 2=majority option, 3=minority option
    df["social_use"] = df["majority_first"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    # y is described as "ID of the site, from 1 to 8"
    df["site"] = df["y"].astype("category")

    # Center age for better interpretability
    df["age_c"] = df["age"] - df["age"].mean()

    print("=== Descriptive statistics ===")
    print("N =", len(df))
    print("Overall choice distribution (1=undemonstrated, 2=majority, 3=minority):")
    print(df["majority_first"].value_counts().sort_index())
    print("\nSocial-use (follow any model) by site:")
    print(df.groupby("site")["social_use"].mean())
    print("\nMajority-choice (vs minority, among social users) by site:")
    print(
        df[df["social_use"] == 1]
        .groupby("site")["majority_choice"]
        .mean()
    )

    print("\nSocial-use by age (quartiles):")
    df["age_bin"] = pd.qcut(df["age"], 4, duplicates="drop")
    print(df.groupby("age_bin")["social_use"].mean())

    print("\nMajority-choice by age (quartiles, among social users):")
    social = df[df["social_use"] == 1].copy()
    social["age_bin"] = pd.qcut(social["age"], 4, duplicates="drop")
    print(social.groupby("age_bin")["majority_choice"].mean())

    print("\n=== Logistic regression: social_use ~ age + site ===")
    model_social = smf.logit("social_use ~ age_c + C(site)", data=df).fit(
        disp=False
    )
    print(model_social.summary())

    print("\n=== Logistic regression: social_use ~ age only ===")
    model_social_age = smf.logit("social_use ~ age_c", data=df).fit(disp=False)
    print(model_social_age.summary())

    print("\n=== Logistic regression: social_use ~ site only ===")
    model_social_site = smf.logit("social_use ~ C(site)", data=df).fit(disp=False)
    print(model_social_site.summary())

    print("\n=== Logistic regression: majority_choice ~ age + site (social users only) ===")
    model_majority = smf.logit(
        "majority_choice ~ age_c + C(site)", data=social
    ).fit(disp=False)
    print(model_majority.summary())

    print("\n=== Logistic regression: majority_choice ~ age only (social users only) ===")
    model_majority_age = smf.logit(
        "majority_choice ~ age_c", data=social
    ).fit(disp=False)
    print(model_majority_age.summary())

    print("\n=== Logistic regression: majority_choice ~ site only (social users only) ===")
    model_majority_site = smf.logit(
        "majority_choice ~ C(site)", data=social
    ).fit(disp=False)
    print(model_majority_site.summary())


if __name__ == "__main__":
    main()

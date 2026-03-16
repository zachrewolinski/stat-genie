import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Outcome recodes
    # 1=unchosen option, 2=majority option, 3=minority option
    df["social_choice"] = df["majority_first"].isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(
        df["majority_first"] == 2,
        1,
        np.where(df["majority_first"] == 3, 0, np.nan),
    )

    # Treat site ID as a categorical proxy for culture
    df["site"] = df["y"].astype("category")

    print("N rows:", len(df))
    print(df.describe(include="all"))

    # Descriptive: social reliance and majority use by age group and site
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 8, 11, 15],
        labels=["4-6", "7-8", "9-11", "12-14"],
        include_lowest=True,
    )

    print("\n=== Proportion using any social information (by age group) ===")
    print(df.groupby("age_group")["social_choice"].mean())

    print("\n=== Proportion using any social information (by site) ===")
    print(df.groupby("site")["social_choice"].mean())

    print("\n=== Proportion choosing majority option (conditional on social use) by age group ===")
    print(
        df[df["majority_choice"].notna()]
        .groupby("age_group")["majority_choice"]
        .mean()
    )

    print("\n=== Proportion choosing majority option (conditional on social use) by site ===")
    print(
        df[df["majority_choice"].notna()]
        .groupby("site")["majority_choice"]
        .mean()
    )

    # Logistic regression: reliance on social information (any demonstrated option vs undemonstrated)
    model_social = smf.logit("social_choice ~ age + C(site) + culture + gender", data=df)
    result_social = model_social.fit(disp=False)
    print("\n=== Logistic model: social_choice ~ age + C(site) + culture + gender ===")
    print(result_social.summary())

    # Logistic regression: preference for majority vs minority, conditional on using social information
    df_social = df[df["majority_choice"].notna()].copy()
    model_majority = smf.logit(
        "majority_choice ~ age + C(site) + culture + gender", data=df_social
    )
    result_majority = model_majority.fit(disp=False)
    print(
        "\n=== Logistic model: majority_choice ~ age + C(site) + culture + gender ==="
    )
    print(result_majority.summary())


if __name__ == "__main__":
    main()

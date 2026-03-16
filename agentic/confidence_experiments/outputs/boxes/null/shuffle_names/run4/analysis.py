import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Recode outcomes
    df["social_choice"] = df["majority_first"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    # Age bands for descriptive summaries
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels)

    print("=== Basic counts ===")
    print(df["majority_first"].value_counts().sort_index())
    print("\n1=undemonstrated, 2=majority, 3=minority\n")

    print("=== Social vs asocial choice rates by age band ===")
    social_by_age = (
        df.groupby("age_band")["social_choice"]
        .mean()
        .rename("social_rate")
        .to_frame()
    )
    print(social_by_age)
    print()

    print("=== Majority-choice rates by age band ===")
    majority_by_age = (
        df.groupby("age_band")["majority_choice"]
        .mean()
        .rename("majority_rate")
        .to_frame()
    )
    print(majority_by_age)
    print()

    print("=== Majority-choice rates by site (y) ===")
    majority_by_site = (
        df.groupby("y")["majority_choice"].mean().rename("majority_rate").to_frame()
    )
    print(majority_by_site)
    print()

    print("=== Social-choice rates by site (y) ===")
    social_by_site = (
        df.groupby("y")["social_choice"].mean().rename("social_rate").to_frame()
    )
    print(social_by_site)
    print()

    # Logistic regression: majority choice ~ age + site + demo order + gender
    print("=== Logistic regression: majority_choice ~ age + C(y) + culture + C(gender) ===")
    model_majority = smf.glm(
        "majority_choice ~ age + C(y) + culture + C(gender)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_majority.summary())
    print()

    # Logistic regression: social choice ~ age + site + demo order + gender
    print("=== Logistic regression: social_choice ~ age + C(y) + culture + C(gender) ===")
    model_social = smf.glm(
        "social_choice ~ age + C(y) + culture + C(gender)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_social.summary())
    print()

    # Interaction: age * site for majority choice to test developmental differences across cultures
    print("=== Logistic regression with interaction: majority_choice ~ age * C(y) + culture + C(gender) ===")
    model_majority_int = smf.glm(
        "majority_choice ~ age * C(y) + culture + C(gender)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_majority_int.summary())


if __name__ == "__main__":
    main()

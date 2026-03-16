import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    print("Head of key variables:")
    print(df[["y", "gender", "age", "majority_first", "culture"]].head())
    print()

    print("Value counts for y (1=unchosen, 2=majority, 3=minority):")
    print(df["y"].value_counts().sort_index())
    print()

    # Derived variables
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    print("Overall reliance on social information (chose majority or minority):")
    print(df["social"].mean())
    print()

    print("Social reliance by culture (row proportions):")
    social_by_culture = pd.crosstab(df["culture"], df["social"], normalize="index")
    print(social_by_culture)
    print()

    print("Social reliance by age (quartiles):")
    df["age_bin"] = pd.qcut(df["age"], q=4, duplicates="drop")
    social_by_age_bin = pd.crosstab(df["age_bin"], df["social"], normalize="index")
    print(social_by_age_bin)
    print()

    # Logistic regression: social reliance as outcome
    print("Logistic regression: social ~ age + culture")
    model_social = smf.glm(
        formula="social ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_social.summary())
    print()

    # Majority preference among those who relied on social information
    df_social = df[df["social"] == 1].copy()
    print("Majority preference rate among social choosers:")
    print(df_social["majority_choice"].mean())
    print()

    print("Majority preference by culture (row proportions):")
    majority_by_culture = pd.crosstab(
        df_social["culture"], df_social["majority_choice"], normalize="index"
    )
    print(majority_by_culture)
    print()

    print("Majority preference by age (quartiles):")
    df_social["age_bin"] = pd.qcut(df_social["age"], q=4, duplicates="drop")
    majority_by_age_bin = pd.crosstab(
        df_social["age_bin"], df_social["majority_choice"], normalize="index"
    )
    print(majority_by_age_bin)
    print()

    print("Logistic regression: majority_choice ~ age + culture (social choosers only)")
    model_majority = smf.glm(
        formula="majority_choice ~ age + C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()
    print(model_majority.summary())


if __name__ == "__main__":
    main()


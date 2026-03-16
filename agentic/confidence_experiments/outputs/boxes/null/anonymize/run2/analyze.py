import pandas as pd
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Encode outcomes
    df["social_choice"] = (df["feature1"] != 1).astype(int)
    df["age"] = df["feature3"]
    df["site"] = df["feature5"].astype("category")

    # Age groups for descriptive summaries
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=True,
    )

    # Majority choice among demonstrated options only
    df_demo = df[df["feature1"] != 1].copy()
    df_demo["majority_choice"] = (df_demo["feature1"] == 2).astype(int)

    print("Overall outcome distribution (1=undemonstrated, 2=majority, 3=minority):")
    print(df["feature1"].value_counts().sort_index())
    print("\nProportions:")
    print(df["feature1"].value_counts(normalize=True).sort_index())

    print("\nReliance on social information (choosing any demonstrated option):")
    print(df["social_choice"].mean())

    print("\nReliance on social information by site:")
    print(df.groupby("site")["social_choice"].mean())

    print("\nReliance on social information by age group:")
    print(df.groupby("age_group")["social_choice"].mean())

    print("\nPreference for majority among social choices (conditional on choosing a demonstrated option):")
    print(df_demo["majority_choice"].mean())

    print("\nPreference for majority among social choices by site:")
    print(df_demo.groupby("site")["majority_choice"].mean())

    print("\nPreference for majority among social choices by age group:")
    print(df_demo.groupby("age_group")["majority_choice"].mean())

    # Logistic models for reliance on social information
    print("\n=== GLM: Social choice ~ age + site ===")
    model_social = glm(
        "social_choice ~ age + C(site)", data=df, family=Binomial()
    ).fit()
    print(model_social.summary())

    print("\n=== GLM: Social choice ~ age * site ===")
    model_social_inter = glm(
        "social_choice ~ age * C(site)", data=df, family=Binomial()
    ).fit()
    print(model_social_inter.summary())

    # Logistic models for majority preference among social choices
    print("\n=== GLM: Majority choice ~ age + site (social choices only) ===")
    model_majority = glm(
        "majority_choice ~ age + C(site)", data=df_demo, family=Binomial()
    ).fit()
    print(model_majority.summary())

    print("\n=== GLM: Majority choice ~ age * site (social choices only) ===")
    model_majority_inter = glm(
        "majority_choice ~ age * C(site)", data=df_demo, family=Binomial()
    ).fit()
    print(model_majority_inter.summary())


if __name__ == "__main__":
    main()


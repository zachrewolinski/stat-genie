import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Encode key outcomes
    df["social_use"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Coarse age groups to allow non-linear developmental patterns
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
    )

    # Model 1: use of any social information (demonstrated options) by age and culture
    model_social = smf.glm(
        formula="social_use ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # Restrict to children who followed a demonstrated option to look at majority preference
    df_social = df[df["social_use"] == 1].copy()

    model_majority = smf.glm(
        formula="majority_choice ~ age + C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    # Print concise summaries to inspect significance patterns
    print("=== Model: Social information use (any demonstrated option) ===")
    print(model_social.summary())
    print("\n=== Model: Majority preference among social learners ===")
    print(model_majority.summary())

    # Alternative specification using age groups
    model_social_ag = smf.glm(
        formula="social_use ~ C(age_group) + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    model_majority_ag = smf.glm(
        formula="majority_choice ~ C(age_group) + C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    print("\n=== Model with age groups: Social information use ===")
    print(model_social_ag.summary())
    print("\n=== Model with age groups: Majority preference ===")
    print(model_majority_ag.summary())

    # Descriptive statistics to aid interpretation
    print("\n=== Descriptive: overall choice proportions ===")
    print(df["y"].value_counts(normalize=True).rename(index={1: "unchosen", 2: "majority", 3: "minority"}))

    print("\n=== Descriptive: majority choice rate by culture ===")
    majority_by_culture = (
        df_social.groupby("culture")["majority_choice"].mean().rename("majority_rate")
    )
    print(majority_by_culture)

    print("\n=== Descriptive: majority choice rate by age group ===")
    majority_by_age_group = (
        df_social.groupby("age_group")["majority_choice"].mean().rename("majority_rate")
    )
    print(majority_by_age_group)


if __name__ == "__main__":
    main()

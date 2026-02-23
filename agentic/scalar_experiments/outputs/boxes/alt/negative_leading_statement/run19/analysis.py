import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived outcomes
    df["social_use"] = (df["y"] != 1).astype(int)
    used_social = df[df["social_use"] == 1].copy()
    used_social["majority_choice"] = (used_social["y"] == 2).astype(int)

    print("=== Descriptive statistics ===")
    print("\nOverall choice distribution (proportions):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nChoice distribution by culture (row-normalized):")
    print(pd.crosstab(df["culture"], df["y"], normalize="index"))

    age_groups = pd.cut(df["age"], bins=[4, 7, 10, 15], right=False)
    print("\nChoice distribution by age group (row-normalized):")
    print(pd.crosstab(age_groups, df["y"], normalize="index"))

    print("\nSocial information use (y != 1) by culture:")
    print(df.groupby("culture")["social_use"].mean())

    print("\nSocial information use by age group:")
    print(df.groupby(age_groups)["social_use"].mean())

    used_social["age_group"] = pd.cut(used_social["age"], bins=[4, 7, 10, 15], right=False)
    print("\nMajority choice (y == 2) among social users by culture:")
    print(used_social.groupby("culture")["majority_choice"].mean())

    print("\nMajority choice among social users by age group:")
    print(used_social.groupby("age_group")["majority_choice"].mean())

    # Logistic regression: reliance on social information
    social_model = smf.logit(
        "social_use ~ age + C(culture) + gender + majority_first", data=df
    ).fit(disp=0)

    social_model_noculture = smf.logit(
        "social_use ~ age + gender + majority_first", data=df
    ).fit(disp=0)

    lr_social_culture = 2 * (social_model.llf - social_model_noculture.llf)
    p_social_culture = chi2.sf(lr_social_culture, df=7)

    # Logistic regression: majority preference among those using social information
    majority_model = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first",
        data=used_social,
    ).fit(disp=0)

    majority_model_noculture = smf.logit(
        "majority_choice ~ age + gender + majority_first", data=used_social
    ).fit(disp=0)

    lr_majority_culture = 2 * (majority_model.llf - majority_model_noculture.llf)
    p_majority_culture = chi2.sf(lr_majority_culture, df=7)

    print("=== Reliance on social information (social_use) ===")
    print(social_model.summary())
    print("\nP-values:\n", social_model.pvalues)

    print(
        "\nLikelihood-ratio test for culture in social_use model: "
        f"LR = {lr_social_culture:.3f}, df = 7, p = {p_social_culture:.3f}"
    )

    print("\n=== Preference for majority option among social users (majority_choice) ===")
    print(majority_model.summary())
    print("\nP-values:\n", majority_model.pvalues)

    print(
        "\nLikelihood-ratio test for culture in majority_choice model: "
        f"LR = {lr_majority_culture:.3f}, df = 7, p = {p_majority_culture:.3f}"
    )


if __name__ == "__main__":
    main()

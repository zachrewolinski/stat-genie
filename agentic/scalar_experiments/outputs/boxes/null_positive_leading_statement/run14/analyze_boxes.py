import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic recoding
    # 1 = undemonstrated option; 2 = majority; 3 = minority
    # Use numeric 0/1 coding for regression.
    df["social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Age bins to look at developmental stages
    bins = [3.5, 6.5, 9.5, 11.5, 14.5]
    labels = ["4-6", "7-9", "10-11", "12-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    print("N =", len(df))
    print("\nOverall outcome distribution (y):")
    print(df["y"].value_counts(normalize=True).rename("proportion"))

    print("\nSocial vs asocial choices overall:")
    print(df["social"].value_counts(normalize=True).rename("proportion"))

    print("\nMajority vs minority among social choices:")
    social_df = df[df["social"] == 1].copy()
    print(social_df["majority_choice"].value_counts(normalize=True).rename("proportion"))

    print("\nSocial choice rate by culture:")
    social_by_culture = (
        df.groupby("culture")["social"]
        .mean()
        .rename("social_rate")
        .to_frame()
    )
    print(social_by_culture)

    print("\nMajority choice (within social) rate by culture:")
    majority_by_culture = (
        social_df.groupby("culture")["majority_choice"]
        .mean()
        .rename("majority_rate")
        .to_frame()
    )
    print(majority_by_culture)

    print("\nSocial choice rate by age group:")
    social_by_age = (
        df.groupby("age_group")["social"]
        .mean()
        .rename("social_rate")
        .to_frame()
    )
    print(social_by_age)

    print("\nMajority choice (within social) rate by age group:")
    majority_by_age = (
        social_df.groupby("age_group")["majority_choice"]
        .mean()
        .rename("majority_rate")
        .to_frame()
    )
    print(majority_by_age)

    # Logistic regression: does social vs asocial vary with age and culture?
    print("\nLogistic regression: social ~ age + culture")
    try:
        df_for_logit = df.dropna(subset=["social", "age", "culture"]).copy()
        model_social = smf.logit("social ~ age + C(culture)", data=df_for_logit).fit(
            disp=False
        )
        print(model_social.summary())
        print("\nPseudo R^2 (McFadden):", model_social.prsquared)
        lr_test = model_social.llr, model_social.llr_pvalue
        print("Likelihood ratio test (chi2, p-value):", lr_test)
    except Exception as e:
        print("Logistic regression for social failed:", e)

    # Logistic regression: majority vs minority/undemonstrated among social choices
    print("\nLogistic regression: majority_choice ~ age + culture (social trials only)")
    try:
        df_for_majority = social_df.dropna(subset=["majority_choice", "age", "culture"]).copy()
        model_majority = smf.logit(
            "majority_choice ~ age + C(culture)", data=df_for_majority
        ).fit(disp=False)
        print(model_majority.summary())
        print("\nPseudo R^2 (McFadden):", model_majority.prsquared)
        lr_test = model_majority.llr, model_majority.llr_pvalue
        print("Likelihood ratio test (chi2, p-value):", lr_test)
    except Exception as e:
        print("Logistic regression for majority failed:", e)


if __name__ == "__main__":
    main()

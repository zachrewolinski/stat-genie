import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic structure
    print("N total:", len(df))
    print("\nOutcome (majority_first) value counts:")
    print(df["majority_first"].value_counts().sort_index())

    print("\nAge summary:")
    print(df["age"].describe())

    print("\nSite (y) counts:")
    print(df["y"].value_counts().sort_index())

    # Derived variables
    df["social_choice"] = df["majority_first"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)

    print("\nSocial choice rate overall:", df["social_choice"].mean())
    print("Majority choice rate overall:", df["majority_choice"].mean())

    print("\nSocial choice rate by site (y):")
    print(df.groupby("y")["social_choice"].mean())

    print("\nMajority choice rate by site (y):")
    print(df.groupby("y")["majority_choice"].mean())

    # Age groups for descriptive purposes
    df["age_group"] = pd.cut(
        df["age"],
        bins=[4, 6, 8, 10, 12, 15],
        right=False,
        labels=["4-5", "6-7", "8-9", "10-11", "12-14"],
    )

    print("\nMajority choice rate by age_group:")
    print(df.groupby("age_group")["majority_choice"].mean())

    print("\nSocial choice rate by age_group:")
    print(df.groupby("age_group")["social_choice"].mean())

    # Logistic regression: reliance on social information
    print("\n--- Logistic regression: social_choice ~ age + C(y) ---")
    glm_social_full = smf.glm(
        "social_choice ~ age + C(y)", data=df, family=sm.families.Binomial()
    ).fit()
    print(glm_social_full.summary())

    glm_social_no_age = smf.glm(
        "social_choice ~ C(y)", data=df, family=sm.families.Binomial()
    ).fit()
    ll_full = glm_social_full.llf
    ll_no_age = glm_social_no_age.llf
    lr_age_stat = 2 * (ll_full - ll_no_age)
    lr_age_df = glm_social_no_age.df_resid - glm_social_full.df_resid
    lr_age_p = chi2.sf(lr_age_stat, lr_age_df)
    print(
        f"\nLR test for age (social_choice): "
        f"chi2={lr_age_stat:.3f}, df={int(lr_age_df)}, p={lr_age_p:.4g}"
    )

    glm_social_no_site = smf.glm(
        "social_choice ~ age", data=df, family=sm.families.Binomial()
    ).fit()
    ll_no_site = glm_social_no_site.llf
    lr_site_stat = 2 * (ll_full - ll_no_site)
    lr_site_df = glm_social_no_site.df_resid - glm_social_full.df_resid
    lr_site_p = chi2.sf(lr_site_stat, lr_site_df)
    print(
        f"LR test for site (C(y)) on social_choice: "
        f"chi2={lr_site_stat:.3f}, df={int(lr_site_df)}, p={lr_site_p:.4g}"
    )

    # Logistic regression: preference for majority cues
    print("\n--- Logistic regression: majority_choice ~ age + C(y) ---")
    glm_maj_full = smf.glm(
        "majority_choice ~ age + C(y)", data=df, family=sm.families.Binomial()
    ).fit()
    print(glm_maj_full.summary())

    glm_maj_no_age = smf.glm(
        "majority_choice ~ C(y)", data=df, family=sm.families.Binomial()
    ).fit()
    ll_maj_full = glm_maj_full.llf
    ll_maj_no_age = glm_maj_no_age.llf
    lr_maj_age_stat = 2 * (ll_maj_full - ll_maj_no_age)
    lr_maj_age_df = glm_maj_no_age.df_resid - glm_maj_full.df_resid
    lr_maj_age_p = chi2.sf(lr_maj_age_stat, lr_maj_age_df)
    print(
        f"\nLR test for age (majority_choice): "
        f"chi2={lr_maj_age_stat:.3f}, df={int(lr_maj_age_df)}, p={lr_maj_age_p:.4g}"
    )

    glm_maj_no_site = smf.glm(
        "majority_choice ~ age", data=df, family=sm.families.Binomial()
    ).fit()
    ll_maj_no_site = glm_maj_no_site.llf
    lr_maj_site_stat = 2 * (ll_maj_full - ll_maj_no_site)
    lr_maj_site_df = glm_maj_no_site.df_resid - glm_maj_full.df_resid
    lr_maj_site_p = chi2.sf(lr_maj_site_stat, lr_maj_site_df)
    print(
        f"LR test for site (C(y)) on majority_choice: "
        f"chi2={lr_maj_site_stat:.3f}, df={int(lr_maj_site_df)}, p={lr_maj_site_p:.4g}"
    )


if __name__ == "__main__":
    main()

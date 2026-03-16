import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    print("Head:")
    print(df.head())
    print("\nValue counts for majority_first (outcome code):")
    print(df["majority_first"].value_counts().sort_index())

    # Recode outcome: 1=undemonstrated/third option, 2=majority, 3=minority
    df["choice_type"] = np.select(
        [
            df["majority_first"] == 1,
            df["majority_first"] == 2,
            df["majority_first"] == 3,
        ],
        ["third", "majority", "minority"],
        default=np.nan,
    )

    # Reliance on social information: chose either majority or minority demonstrator
    df["social"] = df["majority_first"].isin([2, 3]).astype(int)

    # Majority preference among social choices
    social_df = df[df["social"] == 1].copy()
    social_df["majority_choice"] = (social_df["majority_first"] == 2).astype(int)

    # Center age for stability
    df["age_c"] = df["age"] - df["age"].mean()
    social_df["age_c"] = social_df["age"] - social_df["age"].mean()

    # Treat site/culture id as categorical
    df["site"] = df["y"].astype("category")
    social_df["site"] = social_df["y"].astype("category")

    print("\nOverall proportions:")
    print("Social learning rate:", df["social"].mean())
    print("Majority choice rate (among social choices):", social_df["majority_choice"].mean())

    # Logistic regression for reliance on social information
    print("\n=== Logistic regression: social ~ age_c + C(site) ===")
    model_social = smf.glm(
        "social ~ age_c + C(site)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_social.summary())

    # Nested models to test overall effects
    model_social_age_only = smf.glm(
        "social ~ age_c",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    model_social_intercept_only = smf.glm(
        "social ~ 1",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # Likelihood-ratio tests
    lr_site = 2 * (model_social.llf - model_social_age_only.llf)
    df_site = model_social.df_model - model_social_age_only.df_model
    p_site = stats.chi2.sf(lr_site, df_site)

    lr_age = 2 * (model_social_age_only.llf - model_social_intercept_only.llf)
    df_age = model_social_age_only.df_model - model_social_intercept_only.df_model
    p_age = stats.chi2.sf(lr_age, df_age)

    print("\nLR test for site effect on social (controlling age): "
          f"chi2({df_site}) = {lr_site:.2f}, p = {p_site:.3f}")
    print("LR test for age effect on social: "
          f"chi2({df_age}) = {lr_age:.2f}, p = {p_age:.3f}")

    # Logistic regression for majority preference among social choices
    print("\n=== Logistic regression: majority_choice ~ age_c + C(site) (social choices only) ===")
    model_majority = smf.glm(
        "majority_choice ~ age_c + C(site)",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_majority.summary())

    # Nested models for majority preference
    model_majority_age_only = smf.glm(
        "majority_choice ~ age_c",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()
    model_majority_intercept_only = smf.glm(
        "majority_choice ~ 1",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()

    lr_site_m = 2 * (model_majority.llf - model_majority_age_only.llf)
    df_site_m = model_majority.df_model - model_majority_age_only.df_model
    p_site_m = stats.chi2.sf(lr_site_m, df_site_m)

    lr_age_m = 2 * (model_majority_age_only.llf - model_majority_intercept_only.llf)
    df_age_m = model_majority_age_only.df_model - model_majority_intercept_only.df_model
    p_age_m = stats.chi2.sf(lr_age_m, df_age_m)

    print("\nLR test for site effect on majority preference (controlling age): "
          f"chi2({df_site_m}) = {lr_site_m:.2f}, p = {p_site_m:.3f}")
    print("LR test for age effect on majority preference: "
          f"chi2({df_age_m}) = {lr_age_m:.2f}, p = {p_age_m:.3f}")

    # Interaction models: do age effects differ by site?
    print("\n=== Interaction models: age_c * C(site) ===")

    # Social learning interaction
    model_social_int = smf.glm(
        "social ~ age_c * C(site)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    lr_int_social = 2 * (model_social_int.llf - model_social.llf)
    df_int_social = model_social_int.df_model - model_social.df_model
    p_int_social = stats.chi2.sf(lr_int_social, df_int_social)
    print("LR test for age-by-site interaction on social: "
          f"chi2({df_int_social}) = {lr_int_social:.2f}, p = {p_int_social:.3f}")

    # Majority preference interaction
    model_majority_int = smf.glm(
        "majority_choice ~ age_c * C(site)",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()
    lr_int_majority = 2 * (model_majority_int.llf - model_majority.llf)
    df_int_majority = model_majority_int.df_model - model_majority.df_model
    p_int_majority = stats.chi2.sf(lr_int_majority, df_int_majority)
    print("LR test for age-by-site interaction on majority preference: "
          f"chi2({df_int_majority}) = {lr_int_majority:.2f}, p = {p_int_majority:.3f}")


if __name__ == "__main__":
    main()

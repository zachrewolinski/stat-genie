import pandas as pd
from scipy.stats import chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Define outcomes related to the research question
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df_social = df[df["social"] == 1].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Model 1: reliance on social information (any demonstrated option vs undemonstrated)
    model_social_full = smf.glm(
        formula="social ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    model_social_age_only = smf.glm(
        formula="social ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    model_social_culture_only = smf.glm(
        formula="social ~ C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # Likelihood-ratio tests for age and culture
    lr_age_social = 2 * (model_social_full.llf - model_social_culture_only.llf)
    df_age_social = model_social_full.df_model - model_social_culture_only.df_model
    lr_culture_social = 2 * (model_social_full.llf - model_social_age_only.llf)
    df_culture_social = model_social_full.df_model - model_social_age_only.df_model

    p_age_social = chi2.sf(lr_age_social, df_age_social)
    p_culture_social = chi2.sf(lr_culture_social, df_culture_social)

    # Model 2: preference for majority vs minority, among social choosers
    model_maj_full = smf.glm(
        formula="majority_choice ~ age + C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    model_maj_age_only = smf.glm(
        formula="majority_choice ~ age",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    model_maj_culture_only = smf.glm(
        formula="majority_choice ~ C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()

    lr_age_maj = 2 * (model_maj_full.llf - model_maj_culture_only.llf)
    df_age_maj = model_maj_full.df_model - model_maj_culture_only.df_model
    lr_culture_maj = 2 * (model_maj_full.llf - model_maj_age_only.llf)
    df_culture_maj = model_maj_full.df_model - model_maj_age_only.df_model

    p_age_maj = chi2.sf(lr_age_maj, df_age_maj)
    p_culture_maj = chi2.sf(lr_culture_maj, df_culture_maj)

    # Descriptive summaries for interpretation
    by_culture = df.groupby("culture")["social"].mean()
    by_culture_maj = df_social.groupby("culture")["majority_choice"].mean()

    print("=== Reliance on social information (any demonstrated option) ===")
    print("Mean probability of social choice by culture:")
    print(by_culture)
    print("\nLR test for age effect (social vs asocial):")
    print({"stat": lr_age_social, "df": df_age_social, "p_value": p_age_social})
    print("LR test for culture effect (social vs asocial):")
    print({"stat": lr_culture_social, "df": df_culture_social, "p_value": p_culture_social})

    print("\n=== Preference for majority option among social choosers ===")
    print("Mean probability of majority choice by culture:")
    print(by_culture_maj)
    print("\nLR test for age effect (majority vs minority):")
    print({"stat": lr_age_maj, "df": df_age_maj, "p_value": p_age_maj})
    print("LR test for culture effect (majority vs minority):")
    print({"stat": lr_culture_maj, "df": df_culture_maj, "p_value": p_culture_maj})


if __name__ == "__main__":
    main()

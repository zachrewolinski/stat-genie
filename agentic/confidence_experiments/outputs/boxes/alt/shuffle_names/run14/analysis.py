import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Recode outcomes
    # majority_first: 1 = undemonstrated (asocial), 2 = majority option, 3 = minority option
    df["social_choice"] = (df["majority_first"] != 1).astype(int)

    social_df = df[df["social_choice"] == 1].copy()
    social_df["majority_choice"] = (social_df["majority_first"] == 2).astype(int)

    # Treat site ID as categorical proxy for cultural group
    df["site"] = df["y"].astype("category")
    social_df["site"] = social_df["y"].astype("category")

    # Model 1: reliance on social information (any demonstrated option vs undemonstrated)
    social_full = smf.logit("social_choice ~ age + C(site)", data=df).fit(disp=False)
    social_age_only = smf.logit("social_choice ~ age", data=df).fit(disp=False)
    social_site_only = smf.logit("social_choice ~ C(site)", data=df).fit(disp=False)

    # Likelihood-ratio tests for age and site effects
    lr_stat_age_social = 2 * (social_full.llf - social_site_only.llf)
    df_age_social = social_full.df_model - social_site_only.df_model
    p_age_social = stats.chi2.sf(lr_stat_age_social, df_age_social)

    lr_stat_site_social = 2 * (social_full.llf - social_age_only.llf)
    df_site_social = social_full.df_model - social_age_only.df_model
    p_site_social = stats.chi2.sf(lr_stat_site_social, df_site_social)

    # Model 2: preference for majority vs minority among social choices
    majority_full = smf.logit("majority_choice ~ age + C(site)", data=social_df).fit(
        disp=False
    )
    majority_age_only = smf.logit("majority_choice ~ age", data=social_df).fit(
        disp=False
    )
    majority_site_only = smf.logit("majority_choice ~ C(site)", data=social_df).fit(
        disp=False
    )

    lr_stat_age_major = 2 * (majority_full.llf - majority_site_only.llf)
    df_age_major = majority_full.df_model - majority_site_only.df_model
    p_age_major = stats.chi2.sf(lr_stat_age_major, df_age_major)

    lr_stat_site_major = 2 * (majority_full.llf - majority_age_only.llf)
    df_site_major = majority_full.df_model - majority_age_only.df_model
    p_site_major = stats.chi2.sf(lr_stat_site_major, df_site_major)

    # Simple descriptive summaries to aid interpretation
    age_bins = pd.cut(df["age"], bins=[3, 6, 9, 12, 15], labels=["4-6", "7-9", "10-12", "13-14"])
    df["age_group"] = age_bins
    social_df["age_group"] = pd.cut(
        social_df["age"], bins=[3, 6, 9, 12, 15], labels=["4-6", "7-9", "10-12", "13-14"]
    )

    social_by_age = df.groupby("age_group")["social_choice"].mean()
    majority_by_age = social_df.groupby("age_group")["majority_choice"].mean()

    social_by_site = df.groupby("site")["social_choice"].mean()
    majority_by_site = social_df.groupby("site")["majority_choice"].mean()

    print("=== Likelihood-ratio tests ===")
    print(f"Social choice ~ age effect p-value: {p_age_social:.4g}")
    print(f"Social choice ~ site effect p-value: {p_site_social:.4g}")
    print(f"Majority choice ~ age effect p-value: {p_age_major:.4g}")
    print(f"Majority choice ~ site effect p-value: {p_site_major:.4g}")

    print("\n=== Descriptive summaries ===")
    print("Proportion social choice by age group:")
    print(social_by_age)
    print("\nProportion majority (vs minority) among social choices by age group:")
    print(majority_by_age)

    print("\nProportion social choice by site:")
    print(social_by_site)
    print("\nProportion majority (vs minority) among social choices by site:")
    print(majority_by_site)


if __name__ == "__main__":
    main()


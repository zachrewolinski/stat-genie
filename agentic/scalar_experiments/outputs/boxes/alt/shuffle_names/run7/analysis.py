import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def likelihood_ratio_test(model_restricted, model_full, df_diff):
    """Return LR statistic and p-value comparing two nested models."""
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, p_value


def main():
    data_path = Path("boxes.csv")
    if not data_path.exists():
        raise SystemExit("boxes.csv not found")

    df = pd.read_csv(data_path)

    # Rename for clarity based on metadata description
    # majority_first: 1=unchosen option, 2=majority, 3=minority
    df["chose_social"] = (df["majority_first"] != 1).astype(int)
    df["chose_majority"] = (df["majority_first"] == 2).astype(int)

    # Treat site id as a proxy for cultural context
    df["site"] = df["y"].astype("category")

    # Basic descriptive statistics
    n = len(df)
    prop_social_overall = df["chose_social"].mean()
    prop_majority_overall = df["chose_majority"].mean()

    print(f"Total N: {n}")
    print(f"Overall proportion choosing any demonstrated option (social reliance): {prop_social_overall:.3f}")
    print(f"Overall proportion choosing majority option: {prop_majority_overall:.3f}")

    # Summaries by age
    print("\nProportion social / majority choice by age (collapsed across sites):")
    age_summary = (
        df.groupby("age")[["chose_social", "chose_majority"]]
        .mean()
        .rename(columns={"chose_social": "p_social", "chose_majority": "p_majority"})
    )
    print(age_summary)

    # Summaries by site
    print("\nProportion social / majority choice by site:")
    site_summary = (
        df.groupby("site")[["chose_social", "chose_majority"]]
        .mean()
        .rename(columns={"chose_social": "p_social", "chose_majority": "p_majority"})
    )
    print(site_summary)

    # Logistic regression: social reliance ~ age + site + age*site
    print("\n--- Logistic regression: chose_social ---")
    model_social_age = smf.logit("chose_social ~ age", data=df).fit(disp=False)
    model_social_age_site = smf.logit("chose_social ~ age + C(site)", data=df).fit(disp=False)
    model_social_full = smf.logit("chose_social ~ age * C(site)", data=df).fit(disp=False)

    print("\nModel with age only:")
    print(model_social_age.summary2().tables[1])
    print("\nModel with age + site:")
    print(model_social_age_site.summary2().tables[1])
    print("\nFull model with age * site interaction:")
    print(model_social_full.summary2().tables[1])

    # LR tests for additional terms
    lr_site, p_site = likelihood_ratio_test(
        model_social_age, model_social_age_site, df_diff=model_social_age_site.df_model - model_social_age.df_model
    )
    lr_inter, p_inter = likelihood_ratio_test(
        model_social_age_site, model_social_full, df_diff=model_social_full.df_model - model_social_age_site.df_model
    )

    print("\nLR test (adding site to age-only model):")
    print(f"  LR stat = {lr_site:.3f}, p = {p_site:.3g}")
    print("LR test (adding age*site interaction):")
    print(f"  LR stat = {lr_inter:.3f}, p = {p_inter:.3g}")

    # Logistic regression: majority preference among children who used social information
    df_social = df[df["chose_social"] == 1].copy()
    print(f"\nN with social choice (for majority preference analysis): {len(df_social)}")

    print("\n--- Logistic regression: chose_majority (among social choosers) ---")
    model_maj_age = smf.logit("chose_majority ~ age", data=df_social).fit(disp=False)
    model_maj_age_site = smf.logit("chose_majority ~ age + C(site)", data=df_social).fit(disp=False)
    model_maj_full = smf.logit("chose_majority ~ age * C(site)", data=df_social).fit(disp=False)

    print("\nModel with age only:")
    print(model_maj_age.summary2().tables[1])
    print("\nModel with age + site:")
    print(model_maj_age_site.summary2().tables[1])
    print("\nFull model with age * site interaction:")
    print(model_maj_full.summary2().tables[1])

    lr_site_maj, p_site_maj = likelihood_ratio_test(
        model_maj_age, model_maj_age_site, df_diff=model_maj_age_site.df_model - model_maj_age.df_model
    )
    lr_inter_maj, p_inter_maj = likelihood_ratio_test(
        model_maj_age_site, model_maj_full, df_diff=model_maj_full.df_model - model_maj_age_site.df_model
    )

    print("\nLR test (adding site to age-only model, majority preference):")
    print(f"  LR stat = {lr_site_maj:.3f}, p = {p_site_maj:.3g}")
    print("LR test (adding age*site interaction, majority preference):")
    print(f"  LR stat = {lr_inter_maj:.3f}, p = {p_inter_maj:.3g}")

    # Also report simple age-group summaries to connect back to developmental stages
    bins = [4, 6, 9, 12, 14]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True, right=True)
    df_social["age_group"] = pd.cut(df_social["age"], bins=bins, labels=labels, include_lowest=True, right=True)

    print("\nProportion social by age group:")
    print(
        df.groupby("age_group")["chose_social"]
        .mean()
        .to_frame("p_social")
    )

    print("\nProportion majority (among social choosers) by age group:")
    print(
        df_social.groupby("age_group")["chose_majority"]
        .mean()
        .to_frame("p_majority")
    )


if __name__ == "__main__":
    main()


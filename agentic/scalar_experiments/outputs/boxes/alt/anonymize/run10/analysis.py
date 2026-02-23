import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def likelihood_ratio_test(full_model, reduced_model):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Outcome recodes
    # feature1: 1 = undemonstrated third option, 2 = majority option, 3 = minority option
    df["social_reliance"] = df["feature1"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["feature1"] == 2).astype(int)

    # Predictors
    df["age"] = df["feature3"]
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std()
    df["site"] = df["feature5"].astype("category")

    print(f"Number of children: {len(df)}")
    print(f"Sites (feature5) unique values: {sorted(df['site'].unique())}")
    print(f"Age range: {df['age'].min()}–{df['age'].max()}")
    print(f"Overall social reliance rate (chose any demonstrated option): {df['social_reliance'].mean():.3f}")

    social_df = df.copy()

    # Model 1: Reliance on social information (any demonstrated option vs undemonstrated)
    print("\n=== Reliance on social information (social_reliance) ===")
    model_social_base = smf.glm(
        "social_reliance ~ 1",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()
    model_social_age = smf.glm(
        "social_reliance ~ age_z",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()
    model_social_age_site = smf.glm(
        "social_reliance ~ age_z + C(site)",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()
    model_social_age_site_int = smf.glm(
        "social_reliance ~ age_z * C(site)",
        data=social_df,
        family=sm.families.Binomial(),
    ).fit()

    print("\n[Social reliance] Coefficient for age_z (main-effects model):")
    print(model_social_age_site.params.get("age_z"))
    print("[Social reliance] p-value for age_z (main-effects model):")
    print(model_social_age_site.pvalues.get("age_z"))

    lr_age_vs_null, df_age_vs_null, p_age_vs_null = likelihood_ratio_test(
        model_social_age, model_social_base
    )
    print(
        f"[Social reliance] LRT age vs null: chi2({df_age_vs_null:.0f})={lr_age_vs_null:.3f}, p={p_age_vs_null:.3g}"
    )

    lr_site_vs_age, df_site_vs_age, p_site_vs_age = likelihood_ratio_test(
        model_social_age_site, model_social_age
    )
    print(
        f"[Social reliance] LRT site (adding C(site) beyond age): chi2({df_site_vs_age:.0f})={lr_site_vs_age:.3f}, p={p_site_vs_age:.3g}"
    )

    lr_int_vs_main, df_int_vs_main, p_int_vs_main = likelihood_ratio_test(
        model_social_age_site_int, model_social_age_site
    )
    print(
        f"[Social reliance] LRT age × site interaction: chi2({df_int_vs_main:.0f})={lr_int_vs_main:.3f}, p={p_int_vs_main:.3g}"
    )

    # Model 2: Majority preference among children who relied on social information
    social_only = df[df["social_reliance"] == 1].copy()
    print(
        f"\nNumber of children who chose a demonstrated option (for majority/minority analysis): {len(social_only)}"
    )
    print(
        f"Overall probability of choosing majority (among social choices): {social_only['majority_choice'].mean():.3f}"
    )

    print("\n=== Majority preference (majority_choice | social_reliance==1) ===")
    model_maj_base = smf.glm(
        "majority_choice ~ 1",
        data=social_only,
        family=sm.families.Binomial(),
    ).fit()
    model_maj_age = smf.glm(
        "majority_choice ~ age_z",
        data=social_only,
        family=sm.families.Binomial(),
    ).fit()
    model_maj_age_site = smf.glm(
        "majority_choice ~ age_z + C(site)",
        data=social_only,
        family=sm.families.Binomial(),
    ).fit()
    model_maj_age_site_int = smf.glm(
        "majority_choice ~ age_z * C(site)",
        data=social_only,
        family=sm.families.Binomial(),
    ).fit()

    print("\n[Majority preference] Coefficient for age_z (main-effects model):")
    print(model_maj_age_site.params.get("age_z"))
    print("[Majority preference] p-value for age_z (main-effects model):")
    print(model_maj_age_site.pvalues.get("age_z"))

    lr_age_vs_null_maj, df_age_vs_null_maj, p_age_vs_null_maj = likelihood_ratio_test(
        model_maj_age, model_maj_base
    )
    print(
        f"[Majority preference] LRT age vs null: chi2({df_age_vs_null_maj:.0f})={lr_age_vs_null_maj:.3f}, p={p_age_vs_null_maj:.3g}"
    )

    lr_site_vs_age_maj, df_site_vs_age_maj, p_site_vs_age_maj = likelihood_ratio_test(
        model_maj_age_site, model_maj_age
    )
    print(
        f"[Majority preference] LRT site (adding C(site) beyond age): chi2({df_site_vs_age_maj:.0f})={lr_site_vs_age_maj:.3f}, p={p_site_vs_age_maj:.3g}"
    )

    lr_int_vs_main_maj, df_int_vs_main_maj, p_int_vs_main_maj = likelihood_ratio_test(
        model_maj_age_site_int, model_maj_age_site
    )
    print(
        f"[Majority preference] LRT age × site interaction: chi2({df_int_vs_main_maj:.0f})={lr_int_vs_main_maj:.3f}, p={p_int_vs_main_maj:.3g}"
    )


if __name__ == "__main__":
    main()


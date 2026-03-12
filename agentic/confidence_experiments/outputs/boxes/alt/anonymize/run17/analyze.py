import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Rename columns for readability
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Basic recodes
    df["site"] = df["site"].astype("category")
    df["gender"] = df["gender"].astype("category")
    df["majority_first"] = df["majority_first"].astype(int)

    # Reliance on social information: 1 if child chose a demonstrated option (majority or minority)
    df["reliance_social"] = (df["choice"] != 1).astype(int)

    # Majority preference among children who used social information (chose demonstrated option)
    social_df = df[df["choice"].isin([2, 3])].copy()
    social_df["majority_choice"] = (social_df["choice"] == 2).astype(int)

    print("N total:", len(df))
    print("N social (majority or minority):", df["reliance_social"].sum())
    print("Proportion social overall:", df["reliance_social"].mean())
    print()

    print("Reliance on social information by site (mean of reliance_social):")
    print(df.groupby("site")["reliance_social"].mean())
    print()

    print("Reliance on social information by age (mean of reliance_social):")
    print(df.groupby("age")["reliance_social"].mean())
    print()

    print("Majority preference among social users by site (proportion majority_choice):")
    print(social_df.groupby("site")["majority_choice"].mean())
    print()

    print("Majority preference among social users by age (proportion majority_choice):")
    print(social_df.groupby("age")["majority_choice"].mean())
    print()

    # Logistic regression: reliance on social information ~ age + site
    print("Logistic regression for reliance on social information (Logit)")
    formula_reliance_base = "reliance_social ~ age + C(site)"
    model_reliance_base = smf.logit(formula_reliance_base, data=df).fit()
    print(model_reliance_base.summary())
    print()

    # Test whether adding site improves model fit over age-only model
    formula_reliance_age_only = "reliance_social ~ age"
    model_reliance_age_only = smf.logit(formula_reliance_age_only, data=df).fit()
    lr_stat = 2 * (model_reliance_base.llf - model_reliance_age_only.llf)
    lr_df = model_reliance_base.df_model - model_reliance_age_only.df_model
    lr_p = stats.chi2.sf(lr_stat, lr_df)
    print("Reliance on social info: LR test for adding site over age-only model (manual LR test):")
    print(f"  LR stat={lr_stat:.3f}, df={lr_df:.0f}, p={lr_p:.4g}")
    print()

    # Logistic regression with age-by-site interaction for reliance on social information
    print("Logistic regression for reliance on social information with age * site interaction")
    formula_reliance_int = "reliance_social ~ age * C(site)"
    model_reliance_int = smf.logit(formula_reliance_int, data=df).fit()
    lr_stat_int = 2 * (model_reliance_int.llf - model_reliance_base.llf)
    lr_df_int = model_reliance_int.df_model - model_reliance_base.df_model
    lr_p_int = stats.chi2.sf(lr_stat_int, lr_df_int)
    print("Reliance on social info: LR test for adding age * site interaction over main-effects model (manual LR test):")
    print(f"  LR stat={lr_stat_int:.3f}, df={lr_df_int:.0f}, p={lr_p_int:.4g}")
    print()

    # Logistic regression: majority preference among social users
    print("Logistic regression for majority preference among social users (Logit)")
    formula_majority_base = "majority_choice ~ age + C(site) + majority_first"
    model_majority_base = smf.logit(formula_majority_base, data=social_df).fit()
    print(model_majority_base.summary())
    print()

    # Test whether adding site improves model fit over age + majority_first
    formula_majority_reduced = "majority_choice ~ age + majority_first"
    model_majority_reduced = smf.logit(formula_majority_reduced, data=social_df).fit()
    lr_stat2 = 2 * (model_majority_base.llf - model_majority_reduced.llf)
    lr_df2 = model_majority_base.df_model - model_majority_reduced.df_model
    lr_p2 = stats.chi2.sf(lr_stat2, lr_df2)
    print("Majority preference: LR test for adding site (over age + majority_first, manual LR test):")
    print(f"  LR stat={lr_stat2:.3f}, df={lr_df2:.0f}, p={lr_p2:.4g}")
    print()

    # Optional: interaction between age and site for majority preference
    print("Logistic regression for majority preference with age * site interaction")
    formula_majority_int = "majority_choice ~ age * C(site) + majority_first"
    model_majority_int = smf.logit(formula_majority_int, data=social_df).fit()
    lr_stat2_int = 2 * (model_majority_int.llf - model_majority_base.llf)
    lr_df2_int = model_majority_int.df_model - model_majority_base.df_model
    lr_p2_int = stats.chi2.sf(lr_stat2_int, lr_df2_int)
    print("Majority preference: LR test for adding age * site interaction over main-effects model (manual LR test):")
    print(f"  LR stat={lr_stat2_int:.3f}, df={lr_df2_int:.0f}, p={lr_p2_int:.4g}")
    print()


if __name__ == "__main__":
    main()

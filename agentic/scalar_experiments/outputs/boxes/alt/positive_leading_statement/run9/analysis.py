import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def lr_test(full_res, restricted_res):
    lr_stat = 2 * (full_res.llf - restricted_res.llf)
    df_diff = int(full_res.df_model - restricted_res.df_model)
    p_val = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_val


def main():
    df = pd.read_csv("boxes.csv")

    # Basic outcome distribution
    print("Overall outcome counts (1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts().sort_index())
    print()

    # Create derived variables
    df["social"] = (df["y"] != 1).astype(int)
    social_df = df[df["social"] == 1].copy()
    social_df["majority_choice"] = (social_df["y"] == 2).astype(int)

    # Treat culture as categorical
    df["culture"] = df["culture"].astype(int).astype("category")
    social_df["culture"] = social_df["culture"].astype(int).astype("category")

    print("Proportion using any social information (by culture):")
    social_rates_culture = df.groupby("culture")["social"].mean()
    print(social_rates_culture)
    print()

    print("Proportion choosing majority option among social users (by culture):")
    majority_rates_culture = social_df.groupby("culture")["majority_choice"].mean()
    print(majority_rates_culture)
    print()

    # Age groups to get a coarse sense of developmental stages
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    social_df["age_group"] = pd.cut(social_df["age"], bins=bins, labels=labels)

    print("Social information use by age group:")
    print(df.groupby("age_group")["social"].mean())
    print()

    print("Majority preference among social users by age group:")
    print(social_df.groupby("age_group")["majority_choice"].mean())
    print()

    # Logistic regression: social information use ~ age + culture + controls
    print("Logistic regression for social-information use:")
    social_formula_full = "social ~ age + C(culture) + gender + majority_first"
    social_formula_no_culture = "social ~ age + gender + majority_first"
    social_formula_no_age = "social ~ C(culture) + gender + majority_first"

    social_full = smf.logit(social_formula_full, data=df).fit(disp=False)
    social_no_culture = smf.logit(social_formula_no_culture, data=df).fit(disp=False)
    social_no_age = smf.logit(social_formula_no_age, data=df).fit(disp=False)

    print(social_full.summary())
    print()

    lr_culture_social, df_cul_s, p_culture_social = lr_test(social_full, social_no_culture)
    lr_age_social, df_age_s, p_age_social = lr_test(social_full, social_no_age)

    print(f"LR test for culture in social-use model: LR={lr_culture_social:.3f}, df={df_cul_s}, p={p_culture_social:.4g}")
    print(f"LR test for age in social-use model: LR={lr_age_social:.3f}, df={df_age_s}, p={p_age_social:.4g}")
    print()

    # Logistic regression: majority preference among social users
    print("Logistic regression for majority preference among social users:")
    maj_formula_full = "majority_choice ~ age + C(culture) + gender + majority_first"
    maj_formula_no_culture = "majority_choice ~ age + gender + majority_first"
    maj_formula_no_age = "majority_choice ~ C(culture) + gender + majority_first"

    maj_full = smf.logit(maj_formula_full, data=social_df).fit(disp=False)
    maj_no_culture = smf.logit(maj_formula_no_culture, data=social_df).fit(disp=False)
    maj_no_age = smf.logit(maj_formula_no_age, data=social_df).fit(disp=False)

    print(maj_full.summary())
    print()

    lr_culture_maj, df_cul_m, p_culture_maj = lr_test(maj_full, maj_no_culture)
    lr_age_maj, df_age_m, p_age_maj = lr_test(maj_full, maj_no_age)

    print(f"LR test for culture in majority-preference model: LR={lr_culture_maj:.3f}, df={df_cul_m}, p={p_culture_maj:.4g}")
    print(f"LR test for age in majority-preference model: LR={lr_age_maj:.3f}, df={df_age_m}, p={p_age_maj:.4g}")
    print()

    # Also print odds ratios for age for interpretability
    age_coef_social = social_full.params.get("age", np.nan)
    or_age_social = np.exp(age_coef_social) if np.isfinite(age_coef_social) else np.nan
    age_p_social = social_full.pvalues.get("age", np.nan)

    age_coef_maj = maj_full.params.get("age", np.nan)
    or_age_maj = np.exp(age_coef_maj) if np.isfinite(age_coef_maj) else np.nan
    age_p_maj = maj_full.pvalues.get("age", np.nan)

    print(f"Age effect in social-use model: coef={age_coef_social:.3f}, OR={or_age_social:.3f}, p={age_p_social:.4g}")
    print(f"Age effect in majority-preference model: coef={age_coef_maj:.3f}, OR={or_age_maj:.3f}, p={age_p_maj:.4g}")


if __name__ == "__main__":
    main()


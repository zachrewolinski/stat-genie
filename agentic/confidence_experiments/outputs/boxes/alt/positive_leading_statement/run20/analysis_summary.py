import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def lr_test(reduced_model, full_model):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")
    df["social"] = (df["y"] != 1).astype(int)
    df["majority"] = (df["y"] == 2).astype(int)

    print("N observations:", len(df))

    # Chi-square test: outcome distribution varies by culture?
    ctab = pd.crosstab(df["culture"], df["y"])
    chi2, chi_p, chi_df, _ = stats.chi2_contingency(ctab)
    print("\nChi-square test of y by culture:")
    print("chi2 =", round(chi2, 3), "df =", chi_df, "p =", chi_p)

    # Logistic models for social
    print("\n=== Reliance on social information (social) ===")
    social_base = smf.logit(
        "social ~ age + majority_first", data=df
    ).fit(disp=False)
    print(
        "Base model social ~ age + majority_first: "
        f"age coef={social_base.params['age']:.3f}, "
        f"age p={social_base.pvalues['age']:.4g}"
    )

    social_culture = smf.logit(
        "social ~ age + majority_first + C(culture)", data=df
    ).fit(disp=False)
    lr_stat_c, df_c, p_c = lr_test(social_base, social_culture)
    print(
        "Effect of culture on social (LR test): "
        f"LR={lr_stat_c:.3f}, df={df_c}, p={p_c:.4g}"
    )

    social_int = smf.logit(
        "social ~ age * C(culture) + majority_first", data=df
    ).fit(disp=False, maxiter=100)
    lr_stat_int, df_int, p_int = lr_test(social_culture, social_int)
    print(
        "Age*culture interaction on social (LR test): "
        f"LR={lr_stat_int:.3f}, df={df_int}, p={p_int:.4g}"
    )

    # Logistic models for majority
    print("\n=== Preference for majority cues (majority) ===")
    majority_base = smf.logit(
        "majority ~ age + majority_first", data=df
    ).fit(disp=False)
    print(
        "Base model majority ~ age + majority_first: "
        f"age coef={majority_base.params['age']:.3f}, "
        f"age p={majority_base.pvalues['age']:.4g}"
    )

    majority_culture = smf.logit(
        "majority ~ age + majority_first + C(culture)", data=df
    ).fit(disp=False)
    lr_stat_mc, df_mc, p_mc = lr_test(majority_base, majority_culture)
    print(
        "Effect of culture on majority (LR test): "
        f"LR={lr_stat_mc:.3f}, df={df_mc}, p={p_mc:.4g}"
    )

    majority_int = smf.logit(
        "majority ~ age * C(culture) + majority_first", data=df
    ).fit(disp=False, maxiter=100)
    lr_stat_int_m, df_int_m, p_int_m = lr_test(majority_culture, majority_int)
    print(
        "Age*culture interaction on majority (LR test): "
        f"LR={lr_stat_int_m:.3f}, df={df_int_m}, p={p_int_m:.4g}"
    )


if __name__ == "__main__":
    main()


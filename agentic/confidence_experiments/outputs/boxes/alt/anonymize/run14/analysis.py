import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Construct key derived variables
    df["use_social"] = (df["feature1"] != 1).astype(int)
    df["use_majority"] = (df["feature1"] == 2).astype(int)

    # Basic descriptive statistics
    n = len(df)
    use_social_rate = df["use_social"].mean()
    majority_rate = df.loc[df["use_social"] == 1, "use_majority"].mean()

    print(f"Total N: {n}")
    print(f"Overall rate of using social information (majority or minority): {use_social_rate:.3f}")
    print(f"Among social users, rate of choosing majority over minority: {majority_rate:.3f}")

    # Rename columns for clearer formulas
    df = df.rename(
        columns={
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Treat site as categorical
    df["site"] = df["site"].astype("category")

    print("\n=== Logistic regression: reliance on social information (use_social) ===")
    model_social = smf.logit("use_social ~ age + C(site) + gender + majority_first", data=df)
    res_social = model_social.fit(disp=False)
    print(res_social.summary())

    print("\n=== Logistic regression: majority vs minority choice among social users ===")
    df_social = df[df["use_social"] == 1].copy()
    model_majority = smf.logit("use_majority ~ age + C(site) + gender + majority_first", data=df_social)
    res_majority = model_majority.fit(disp=False)
    print(res_majority.summary())

    # Simple effect-size style summaries for later interpretation
    print("\n=== Age effects (odds ratios) ===")
    if "age" in res_social.params.index:
        or_age_social = np.exp(res_social.params["age"])
        p_age_social = res_social.pvalues["age"]
        print(f"Social reliance: OR per year = {or_age_social:.3f}, p = {p_age_social:.4g}")
    if "age" in res_majority.params.index:
        or_age_majority = np.exp(res_majority.params["age"])
        p_age_majority = res_majority.pvalues["age"]
        print(f"Majority preference: OR per year = {or_age_majority:.3f}, p = {p_age_majority:.4g}")

    # Site effects via likelihood ratio tests
    print("\n=== Likelihood-ratio tests for site effects ===")
    model_social_nosite = smf.logit("use_social ~ age + gender + majority_first", data=df)
    res_social_nosite = model_social_nosite.fit(disp=False)
    lr_social = 2 * (res_social.llf - res_social_nosite.llf)
    df_social_lr = res_social.df_model - res_social_nosite.df_model
    p_social_lr = chi2.sf(lr_social, df_social_lr)
    print(
        f"Social reliance: LR statistic = {lr_social:.3f}, df = {df_social_lr}, "
        f"p = {p_social_lr:.4g}"
    )

    model_majority_nosite = smf.logit("use_majority ~ age + gender + majority_first", data=df_social)
    res_majority_nosite = model_majority_nosite.fit(disp=False)
    lr_majority = 2 * (res_majority.llf - res_majority_nosite.llf)
    df_majority_lr = res_majority.df_model - res_majority_nosite.df_model
    p_majority_lr = chi2.sf(lr_majority, df_majority_lr)
    print(
        f"Majority preference: LR statistic = {lr_majority:.3f}, df = {df_majority_lr}, "
        f"p = {p_majority_lr:.4g}"
    )


if __name__ == "__main__":
    main()

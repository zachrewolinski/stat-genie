import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_logit(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data)
    result = model.fit(disp=False)
    return result


def lr_test(full_result, reduced_result, df_diff: int):
    lr_stat = 2 * (full_result.llf - reduced_result.llf)
    p_value = chi2.sf(lr_stat, df_diff)
    return lr_stat, p_value


def main():
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Basic recoding
    df["social"] = df["feature1"].isin([2, 3]).astype(int)
    df["majority_choice"] = df["feature1"].map({2: 1, 3: 0})
    df["age"] = df["feature3"].astype(float)
    df["site"] = df["feature5"].astype(int)

    n = len(df)
    n_sites = df["site"].nunique()
    age_min, age_max = df["age"].min(), df["age"].max()

    # Descriptive summaries
    social_rate = df["social"].mean()
    majority_rate = df["majority_choice"].mean()

    # Age-grouped summaries
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    age_group_summary = (
        df.groupby("age_group")
        .agg(
            n=("social", "size"),
            social_rate=("social", "mean"),
            majority_rate=("majority_choice", "mean"),
        )
        .reset_index()
    )

    site_summary = (
        df.groupby("site")
        .agg(
            n=("social", "size"),
            social_rate=("social", "mean"),
            majority_rate=("majority_choice", "mean"),
        )
        .reset_index()
    )

    # Logistic regression: reliance on social information (any demonstrated option)
    logit_social_full = fit_logit(
        "social ~ age + C(feature5) + feature2 + feature4", df
    )
    logit_social_age_only = fit_logit("social ~ age", df)
    logit_social_no_site = fit_logit("social ~ age + feature2 + feature4", df)

    # LR test for site (culture) effect on social reliance
    df_diff_site_social = (
        logit_social_full.df_model - logit_social_no_site.df_model
    )
    lr_social_site, p_social_site = lr_test(
        logit_social_full, logit_social_no_site, int(df_diff_site_social)
    )

    # Logistic regression: majority preference given social learning
    df_social = df[df["majority_choice"].notna()].copy()
    logit_majority_full = fit_logit(
        "majority_choice ~ age + C(feature5) + feature2 + feature4", df_social
    )
    logit_majority_age_only = fit_logit("majority_choice ~ age", df_social)
    logit_majority_no_site = fit_logit(
        "majority_choice ~ age + feature2 + feature4", df_social
    )

    # LR test for site (culture) effect on majority preference
    df_diff_site_majority = (
        logit_majority_full.df_model - logit_majority_no_site.df_model
    )
    lr_majority_site, p_majority_site = lr_test(
        logit_majority_full, logit_majority_no_site, int(df_diff_site_majority)
    )

    # Age effects (Wald tests on age coefficient)
    age_coef_social = logit_social_full.params["age"]
    age_p_social = logit_social_full.pvalues["age"]

    age_coef_majority = logit_majority_full.params["age"]
    age_p_majority = logit_majority_full.pvalues["age"]

    # Print a concise summary for manual inspection
    print("N observations:", n)
    print("Number of sites:", n_sites)
    print(f"Age range: {age_min:.1f}–{age_max:.1f}")
    print(f"Overall social-learning rate: {social_rate:.3f}")
    print(f"Overall majority-choice rate (among social learners): {majority_rate:.3f}")
    print("\nAge-grouped summary:")
    print(age_group_summary.to_string(index=False))
    print("\nSite summary:")
    print(site_summary.to_string(index=False))

    print("\nLogit: Social (any demonstrated option)")
    print("Age coefficient:", age_coef_social, "p=", age_p_social)
    print(
        "LR test for site effect on social: LR=",
        lr_social_site,
        "p=",
        p_social_site,
    )

    print("\nLogit: Majority preference (vs minority, among social learners)")
    print("Age coefficient:", age_coef_majority, "p=", age_p_majority)
    print(
        "LR test for site effect on majority preference: LR=",
        lr_majority_site,
        "p=",
        p_majority_site,
    )


if __name__ == "__main__":
    main()

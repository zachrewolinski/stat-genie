import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Construct key derived variables
    df["social"] = (df["y"] != 1).astype(int)  # 1 = followed any demonstrated option
    df["majority_choice"] = np.where(
        df["y"] == 2,
        1,
        np.where(df["y"] == 3, 0, np.nan),
    )  # among social choices, 1 = majority, 0 = minority

    df["culture"] = df["culture"].astype("category")

    print("=== Basic counts ===")
    print("N rows:", len(df))
    print("Outcome distribution (y):")
    print(df["y"].value_counts().sort_index())
    print("Outcome distribution (proportions):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\n=== Reliance on social information (any demonstrated option) ===")
    print("Overall social-choice rate:", df["social"].mean())
    print("\nSocial-choice rate by culture:")
    print(df.groupby("culture")["social"].mean())

    # Age groups for descriptive summaries
    age_bins = [4, 6, 8, 10, 12, 14]
    age_labels = ["4-5", "6-7", "8-9", "10-11", "12-13"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
    print("\nSocial-choice rate by age group:")
    print(df.groupby("age_group")["social"].mean())

    print("\n=== Logistic regression: social ~ age + culture ===")
    model_social = smf.glm(
        "social ~ age + C(culture)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model_social.summary())
    print("\nCoefficients for age and cultures (social):")
    print(model_social.params)
    print("\nP-values (social):")
    print(model_social.pvalues)

    # Likelihood-ratio test for the overall contribution of culture
    model_social_nocult = smf.glm(
        "social ~ age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    lr_stat_social = 2 * (model_social.llf - model_social_nocult.llf)
    df_diff_social = model_social.df_model - model_social_nocult.df_model
    p_lr_social = stats.chi2.sf(lr_stat_social, df_diff_social)
    print("\nLikelihood-ratio test for adding culture to social model:")
    print(f"LR stat = {lr_stat_social:.3f}, df = {df_diff_social}, p = {p_lr_social:.4f}")

    print("\n=== Majority preference among social choosers ===")
    df_social = df[df["social"] == 1].copy()
    print("N social choosers:", len(df_social))
    print("Overall majority-choice rate among social choosers:", df_social["majority_choice"].mean())
    print("\nMajority-choice rate by culture:")
    print(df_social.groupby("culture")["majority_choice"].mean())
    print("\nMajority-choice rate by age group:")
    print(df_social.groupby("age_group")["majority_choice"].mean())

    print("\n=== Logistic regression: majority_choice ~ age + culture (among social choosers) ===")
    model_majority = smf.glm(
        "majority_choice ~ age + C(culture)",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()
    print(model_majority.summary())
    print("\nCoefficients (majority-choice model):")
    print(model_majority.params)
    print("\nP-values (majority-choice model):")
    print(model_majority.pvalues)

    # Likelihood-ratio test for the overall contribution of culture
    model_majority_nocult = smf.glm(
        "majority_choice ~ age",
        data=df_social,
        family=sm.families.Binomial(),
    ).fit()
    lr_stat_maj = 2 * (model_majority.llf - model_majority_nocult.llf)
    df_diff_maj = model_majority.df_model - model_majority_nocult.df_model
    p_lr_maj = stats.chi2.sf(lr_stat_maj, df_diff_maj)
    print("\nLikelihood-ratio test for adding culture to majority-choice model:")
    print(f"LR stat = {lr_stat_maj:.3f}, df = {df_diff_maj}, p = {p_lr_maj:.4f}")


if __name__ == "__main__":
    main()

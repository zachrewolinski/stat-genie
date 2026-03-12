import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def main() -> None:
    cwd = Path(__file__).parent
    data_path = cwd / "boxes.csv"

    df = pd.read_csv(data_path)

    # Basic sanity checks
    print("N rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df.describe(include="all"))

    # Create derived outcomes
    df["social_choice"] = (df["y"] != 1).astype(int)
    df_social = df.copy()

    df_majority = df[df["y"].isin([2, 3])].copy()
    df_majority["majority_choice"] = (df_majority["y"] == 2).astype(int)

    # Center age for stability
    df_social["age_c"] = df_social["age"] - df_social["age"].mean()
    df_majority["age_c"] = df_majority["age"] - df_majority["age"].mean()

    # Model 1: reliance on social information (any demonstrated option vs undemonstrated)
    print("\n=== Logistic regression: social_choice ~ age + culture + gender + majority_first ===")
    formula_social = "social_choice ~ age_c + C(culture) + gender + majority_first"
    model_social = smf.logit(formula_social, data=df_social).fit(disp=False)
    print(model_social.summary())

    # Likelihood ratio test for joint effect of culture in social_choice model
    reduced_social = smf.logit("social_choice ~ age_c + gender + majority_first", data=df_social).fit(disp=False)
    lr_stat_social = 2 * (model_social.llf - reduced_social.llf)
    df_diff_social = model_social.df_model - reduced_social.df_model
    p_lr_social = chi2.sf(lr_stat_social, df_diff_social)
    print("\nLR test for culture (social_choice):")
    print("  LR stat =", lr_stat_social, "df =", df_diff_social, "p =", p_lr_social)

    # Model 2: majority vs minority among social choosers
    print("\n=== Logistic regression: majority_choice ~ age + culture + gender + majority_first ===")
    formula_majority = "majority_choice ~ age_c + C(culture) + gender + majority_first"
    model_majority = smf.logit(formula_majority, data=df_majority).fit(disp=False)
    print(model_majority.summary())

    reduced_majority = smf.logit("majority_choice ~ age_c + gender + majority_first", data=df_majority).fit(
        disp=False
    )
    lr_stat_majority = 2 * (model_majority.llf - reduced_majority.llf)
    df_diff_majority = model_majority.df_model - reduced_majority.df_model
    p_lr_majority = chi2.sf(lr_stat_majority, df_diff_majority)
    print("\nLR test for culture (majority_choice):")
    print("  LR stat =", lr_stat_majority, "df =", df_diff_majority, "p =", p_lr_majority)

    # Simple descriptive summaries by age and culture for interpretability
    df["age_group"] = pd.cut(df["age"], bins=[3, 6, 9, 12, 15], labels=["4-6", "7-9", "10-12", "13-14"])
    df_majority["age_group"] = pd.cut(
        df_majority["age"], bins=[3, 6, 9, 12, 15], labels=["4-6", "7-9", "10-12", "13-14"]
    )

    social_rate = (
        df.groupby(["culture", "age_group"])["social_choice"]
        .mean()
        .unstack()
        .sort_index()
    )
    print("\nMean social-choice rate by culture and age_group:")
    print(social_rate)

    majority_rate = (
        df_majority.groupby(["culture", "age_group"])["majority_choice"]
        .mean()
        .unstack()
        .sort_index()
    )
    print("\nMean majority-choice rate (conditional on social choice) by culture and age_group:")
    print(majority_rate)


if __name__ == "__main__":
    main()

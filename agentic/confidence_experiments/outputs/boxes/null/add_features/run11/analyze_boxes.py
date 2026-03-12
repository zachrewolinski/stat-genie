import pandas as pd
import numpy as np
from scipy.stats import chi2
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic structure
    print("N observations:", len(df))
    print("Columns:", list(df.columns))

    # Outcome coding
    df["social_choice"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    print("\nOutcome distribution (y):")
    print(df["y"].value_counts().sort_index())
    print("\nOutcome distribution (y, proportion):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nSocial learning rate (any demonstrated option):", df["social_choice"].mean())
    print("Majority choice rate (overall):", df["majority_choice"].mean())

    df_social = df[df["social_choice"] == 1].copy()
    print("\nN social learners:", len(df_social))
    print("Majority choice rate among social learners:", df_social["majority_choice"].mean())

    # Helper for LR tests (manual, since LogitResults lacks compare_lr_test)
    def lr_test(full_res, reduced_res, label: str) -> None:
        lr_stat = 2.0 * (full_res.llf - reduced_res.llf)
        df_diff = int(full_res.df_model - reduced_res.df_model)
        if df_diff <= 0:
            print(f"\n[Warning] Non-positive df_diff for {label}; skipping LR test.")
            return
        p_value = chi2.sf(lr_stat, df_diff)
        print(f"\nLR test for {label}:")
        print(f"  LR stat = {lr_stat:.3f}, df = {df_diff}, p = {p_value:.5f}")

    # Model 1: reliance on social information (social_choice)
    print("\n=== Logistic model: social_choice ~ age + culture + gender + majority_first ===")
    formula_social_full = "social_choice ~ age + C(culture) + gender + majority_first"
    formula_social_no_age = "social_choice ~ C(culture) + gender + majority_first"
    formula_social_no_culture = "social_choice ~ age + gender + majority_first"

    social_full = smf.logit(formula_social_full, data=df).fit(disp=False, maxiter=500)
    social_no_age = smf.logit(formula_social_no_age, data=df).fit(disp=False, maxiter=500)
    social_no_culture = smf.logit(formula_social_no_culture, data=df).fit(disp=False, maxiter=500)

    print("\nSocial-choice model summary (full):")
    print(social_full.summary())

    lr_test(social_full, social_no_age, "age effect on social_choice")
    lr_test(social_full, social_no_culture, "culture effect on social_choice")

    # Predicted probabilities for social_choice by age and culture
    print("\nPredicted probability of social learning (social_choice=1):")
    mode_gender = int(df["gender"].mode()[0])
    mode_mf = int(df["majority_first"].mode()[0])
    mode_culture = int(df["culture"].mode()[0])

    age_quantiles = np.percentile(df["age"], [25, 50, 75])
    print("\nBy age (holding culture, gender, majority_first at typical values):")
    for a in age_quantiles:
        pred_df = pd.DataFrame(
            {
                "age": [a],
                "culture": [mode_culture],
                "gender": [mode_gender],
                "majority_first": [mode_mf],
            }
        )
        prob = float(social_full.predict(pred_df)[0])
        print(f"  Age {a:.1f}: P(social_choice=1) ~ {prob:.3f}")

    cultures = sorted(df["culture"].unique())
    print("\nBy culture (holding age, gender, majority_first at typical values):")
    base_age = float(df["age"].mean())
    for c in cultures:
        pred_df = pd.DataFrame(
            {
                "age": [base_age],
                "culture": [c],
                "gender": [mode_gender],
                "majority_first": [mode_mf],
            }
        )
        prob = float(social_full.predict(pred_df)[0])
        print(f"  Culture {c}: P(social_choice=1) ~ {prob:.3f}")

    # Model 2: majority preference among social learners (majority_choice)
    print("\n=== Logistic model: majority_choice (among social learners) ~ age + culture + gender + majority_first ===")
    formula_majority_full = "majority_choice ~ age + C(culture) + gender + majority_first"
    formula_majority_no_age = "majority_choice ~ C(culture) + gender + majority_first"
    formula_majority_no_culture = "majority_choice ~ age + gender + majority_first"

    majority_full = smf.logit(formula_majority_full, data=df_social).fit(disp=False, maxiter=500)
    majority_no_age = smf.logit(formula_majority_no_age, data=df_social).fit(disp=False, maxiter=500)
    majority_no_culture = smf.logit(formula_majority_no_culture, data=df_social).fit(disp=False, maxiter=500)

    print("\nMajority-choice model summary (full):")
    print(majority_full.summary())

    lr_test(majority_full, majority_no_age, "age effect on majority_choice")
    lr_test(majority_full, majority_no_culture, "culture effect on majority_choice")

    print("\nPredicted probability of majority choice (majority_choice=1) among social learners:")
    age_quantiles_social = np.percentile(df_social["age"], [25, 50, 75])
    print("\nBy age (holding culture, gender, majority_first at typical values):")
    mode_gender_social = int(df_social["gender"].mode()[0])
    mode_mf_social = int(df_social["majority_first"].mode()[0])
    mode_culture_social = int(df_social["culture"].mode()[0])

    for a in age_quantiles_social:
        pred_df = pd.DataFrame(
            {
                "age": [a],
                "culture": [mode_culture_social],
                "gender": [mode_gender_social],
                "majority_first": [mode_mf_social],
            }
        )
        prob = float(majority_full.predict(pred_df)[0])
        print(f"  Age {a:.1f}: P(majority_choice=1) ~ {prob:.3f}")

    cultures_social = sorted(df_social["culture"].unique())
    print("\nBy culture (holding age, gender, majority_first at typical values):")
    base_age_social = float(df_social["age"].mean())
    for c in cultures_social:
        pred_df = pd.DataFrame(
            {
                "age": [base_age_social],
                "culture": [c],
                "gender": [mode_gender_social],
                "majority_first": [mode_mf_social],
            }
        )
        prob = float(majority_full.predict(pred_df)[0])
        print(f"  Culture {c}: P(majority_choice=1) ~ {prob:.3f}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Define key derived outcomes
    df["social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Age groups to approximate developmental stages
    bins = [3, 6, 9, 12, 14]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    # Descriptive summaries
    print("=== Descriptive summaries by culture and age_group ===")
    desc = (
        df.groupby(["culture", "age_group"])
        .agg(
            n=("y", "size"),
            social_rate=("social", "mean"),
            majority_rate=("majority_choice", "mean"),
        )
        .reset_index()
    )
    print(desc.to_string(index=False))

    # Majority preference among children who relied on social information
    df_social = df[df["social"] == 1].copy()
    maj_social = (
        df_social.groupby(["culture", "age_group"])
        .agg(
            n=("y", "size"),
            majority_given_social=("majority_choice", "mean"),
        )
        .reset_index()
    )
    print("\n=== Majority preference among social choosers ===")
    print(maj_social.to_string(index=False))

    # Logistic regression: reliance on social information
    print("\n=== Logistic regression: social ~ age + culture + gender + majority_first ===")
    model_social = smf.logit(
        "social ~ age + C(culture) + gender + majority_first", data=df
    ).fit(disp=False, maxiter=200)
    print(model_social.summary())

    # LR test for culture effect on social reliance
    model_social_noculture = smf.logit(
        "social ~ age + gender + majority_first", data=df
    ).fit(disp=False, maxiter=200)
    lr_stat_social = 2 * (model_social.llf - model_social_noculture.llf)
    df_diff_social = model_social.df_model - model_social_noculture.df_model
    pval_social_culture = stats.chi2.sf(lr_stat_social, df_diff_social)
    print(
        f"\nLR test for culture effect on social reliance: "
        f"chi2={lr_stat_social:.3f}, df={df_diff_social:.0f}, p={pval_social_culture:.4g}"
    )

    age_coef_social = model_social.params["age"]
    age_p_social = model_social.pvalues["age"]
    print(
        f"Age effect on social reliance: coef={age_coef_social:.3f}, p={age_p_social:.4g}"
    )

    # Logistic regression: majority preference among social choosers
    print(
        "\n=== Logistic regression: majority_choice ~ age + culture + gender + majority_first (social choosers only) ==="
    )
    model_majority = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first", data=df_social
    ).fit(disp=False, maxiter=200)
    print(model_majority.summary())

    model_majority_noculture = smf.logit(
        "majority_choice ~ age + gender + majority_first", data=df_social
    ).fit(disp=False, maxiter=200)
    lr_stat_majority = 2 * (model_majority.llf - model_majority_noculture.llf)
    df_diff_majority = model_majority.df_model - model_majority_noculture.df_model
    pval_majority_culture = stats.chi2.sf(lr_stat_majority, df_diff_majority)
    print(
        f"\nLR test for culture effect on majority preference: "
        f"chi2={lr_stat_majority:.3f}, df={df_diff_majority:.0f}, p={pval_majority_culture:.4g}"
    )

    age_coef_majority = model_majority.params["age"]
    age_p_majority = model_majority.pvalues["age"]
    print(
        f"Age effect on majority preference: coef={age_coef_majority:.3f}, p={age_p_majority:.4g}"
    )


if __name__ == "__main__":
    main()


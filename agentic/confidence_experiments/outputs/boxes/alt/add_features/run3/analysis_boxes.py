import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived outcomes
    df["social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    n = len(df)
    social_rate = df["social"].mean()
    df_social_only = df[df["social"] == 1].copy()
    majority_rate_overall = df_social_only["majority_choice"].mean()

    # Variation in raw proportions across cultures
    social_by_culture = df.groupby("culture")["social"].mean()
    majority_by_culture = df_social_only.groupby("culture")["majority_choice"].mean()

    print(f"Sample size (children): {n}")
    print(f"Overall probability of using social information (y != 1): {social_rate:.3f}")
    print(
        "Overall probability of choosing the majority option",
        "conditional on using social information (y != 1):",
        f"{majority_rate_overall:.3f}",
    )
    print(
        "Social-information use by culture (min, max):",
        f"{social_by_culture.min():.3f}",
        f"{social_by_culture.max():.3f}",
    )
    print(
        "Majority-choice rate by culture among social learners (min, max):",
        f"{majority_by_culture.min():.3f}",
        f"{majority_by_culture.max():.3f}",
    )

    # Logistic model: reliance on social information (any demonstrated option vs undemonstrated)
    model_social_full = smf.logit(
        "social ~ age + C(culture) + majority_first + gender", data=df
    ).fit(disp=False)
    model_social_reduced = smf.logit(
        "social ~ age + majority_first + gender", data=df
    ).fit(disp=False)

    lr_stat_social_culture = 2 * (model_social_full.llf - model_social_reduced.llf)
    df_social_culture = model_social_full.df_model - model_social_reduced.df_model
    p_social_culture = stats.chi2.sf(lr_stat_social_culture, df_social_culture)
    p_social_age = float(model_social_full.pvalues["age"])

    # Logistic model: majority vs minority choice among children who used social information
    if df_social_only["majority_choice"].nunique() > 1:
        model_majority_full = smf.logit(
            "majority_choice ~ age + C(culture) + majority_first + gender",
            data=df_social_only,
        ).fit(disp=False)
        model_majority_reduced = smf.logit(
            "majority_choice ~ age + majority_first + gender",
            data=df_social_only,
        ).fit(disp=False)

        lr_stat_majority_culture = 2 * (
            model_majority_full.llf - model_majority_reduced.llf
        )
        df_majority_culture = (
            model_majority_full.df_model - model_majority_reduced.df_model
        )
        p_majority_culture = stats.chi2.sf(
            lr_stat_majority_culture, df_majority_culture
        )
        p_majority_age = float(model_majority_full.pvalues["age"])
    else:
        model_majority_full = None
        p_majority_culture = np.nan
        p_majority_age = np.nan

    # Effect size: change in predicted probabilities from lower to higher age
    age_low, age_high = df["age"].quantile([0.1, 0.9])
    ref_culture = df["culture"].value_counts().idxmax()
    base_row = {
        "age": age_low,
        "culture": ref_culture,
        "majority_first": df["majority_first"].mode().iloc[0],
        "gender": df["gender"].mode().iloc[0],
    }

    df_pred_social = pd.DataFrame(
        [base_row, {**base_row, "age": age_high}],
    )
    pred_social = model_social_full.predict(df_pred_social)
    social_diff = float(pred_social.iloc[1] - pred_social.iloc[0])

    if model_majority_full is not None:
        df_pred_majority = pd.DataFrame(
            [base_row, {**base_row, "age": age_high}],
        )
        pred_majority = model_majority_full.predict(df_pred_majority)
        majority_diff = float(pred_majority.iloc[1] - pred_majority.iloc[0])
    else:
        majority_diff = np.nan

    print(
        "Social learning:",
        "p_age=",
        p_social_age,
        "p_culture=",
        p_social_culture,
        "pred_prob_diff_age=",
        social_diff,
    )
    print(
        "Majority preference:",
        "p_age=",
        p_majority_age,
        "p_culture=",
        p_majority_culture,
        "pred_prob_diff_age=",
        majority_diff,
    )


if __name__ == "__main__":
    main()

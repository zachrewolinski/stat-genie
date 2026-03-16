import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def lr_test(model_restricted, model_full):
    """Likelihood-ratio test comparing two nested models."""
    lr = 2.0 * (model_full.llf - model_restricted.llf)
    df = model_full.df_model - model_restricted.df_model
    p = stats.chi2.sf(lr, df)
    return float(lr), int(df), float(p)


def main():
    df = pd.read_csv("boxes.csv")

    # Recode outcomes for the two key constructs.
    # Reliance on social information: choosing any demonstrated option vs the undemonstrated one.
    df["social_choice"] = np.where(df["y"].isin([2, 3]), 1, 0)

    # Majority preference: choosing the majority option vs minority/other.
    df["majority_choice"] = np.where(df["y"] == 2, 1, 0)

    # Majority vs minority among those who relied on social information.
    df_social = df[df["social_choice"] == 1].copy()
    df_social["majority_vs_minority"] = np.where(df_social["y"] == 2, 1, 0)

    # Center age for stability in interaction models.
    df["age_c"] = df["age"] - df["age"].mean()
    df_social["age_c"] = df_social["age"] - df_social["age"].mean()

    # --- Models for reliance on social information ---
    m_social_intercept = smf.logit("social_choice ~ 1", data=df).fit(disp=False)
    m_social_age = smf.logit("social_choice ~ age_c", data=df).fit(disp=False)
    m_social_age_culture = smf.logit("social_choice ~ age_c + C(culture)", data=df).fit(disp=False)
    m_social_full = smf.logit("social_choice ~ age_c * C(culture)", data=df).fit(disp=False)

    social_age_lr = lr_test(m_social_intercept, m_social_age)
    social_culture_lr = lr_test(m_social_age, m_social_age_culture)
    social_interaction_lr = lr_test(m_social_age_culture, m_social_full)

    # --- Models for majority preference among social choosers ---
    m_maj_intercept = smf.logit("majority_vs_minority ~ 1", data=df_social).fit(disp=False)
    m_maj_age = smf.logit("majority_vs_minority ~ age_c", data=df_social).fit(disp=False)
    m_maj_age_culture = smf.logit("majority_vs_minority ~ age_c + C(culture)", data=df_social).fit(
        disp=False
    )
    m_maj_full = smf.logit(
        "majority_vs_minority ~ age_c * C(culture)", data=df_social
    ).fit(disp=False)

    maj_age_lr = lr_test(m_maj_intercept, m_maj_age)
    maj_culture_lr = lr_test(m_maj_age, m_maj_age_culture)
    maj_interaction_lr = lr_test(m_maj_age_culture, m_maj_full)

    # Descriptive summaries by age group (quartiles) and culture.
    df["age_group"] = pd.qcut(df["age"], q=4, duplicates="drop")
    df_social["age_group"] = pd.qcut(df_social["age"], q=4, duplicates="drop")

    social_by_culture = (
        df.groupby("culture")["social_choice"].mean().rename("social_rate").reset_index()
    )
    majority_by_culture = (
        df.groupby("culture")["majority_choice"].mean().rename("majority_rate").reset_index()
    )

    social_by_age = (
        df.groupby("age_group")["social_choice"]
        .mean()
        .rename("social_rate")
        .reset_index()
    )
    social_by_age["age_group"] = social_by_age["age_group"].astype(str)
    majority_by_age = (
        df.groupby("age_group")["majority_choice"]
        .mean()
        .rename("majority_rate")
        .reset_index()
    )
    majority_by_age["age_group"] = majority_by_age["age_group"].astype(str)

    maj_bias_by_culture = (
        df_social.groupby("culture")["majority_vs_minority"]
        .mean()
        .rename("majority_among_social")
        .reset_index()
    )
    maj_bias_by_age = (
        df_social.groupby("age_group")["majority_vs_minority"]
        .mean()
        .rename("majority_among_social")
        .reset_index()
    )
    maj_bias_by_age["age_group"] = maj_bias_by_age["age_group"].astype(str)

    results = {
        "n": int(df.shape[0]),
        "social_choice_rate": float(df["social_choice"].mean()),
        "majority_choice_rate": float(df["majority_choice"].mean()),
        "lr_tests": {
            "social_age": {
                "lr_stat": social_age_lr[0],
                "df": social_age_lr[1],
                "p_value": social_age_lr[2],
            },
            "social_culture": {
                "lr_stat": social_culture_lr[0],
                "df": social_culture_lr[1],
                "p_value": social_culture_lr[2],
            },
            "social_age_x_culture": {
                "lr_stat": social_interaction_lr[0],
                "df": social_interaction_lr[1],
                "p_value": social_interaction_lr[2],
            },
            "majority_age": {
                "lr_stat": maj_age_lr[0],
                "df": maj_age_lr[1],
                "p_value": maj_age_lr[2],
            },
            "majority_culture": {
                "lr_stat": maj_culture_lr[0],
                "df": maj_culture_lr[1],
                "p_value": maj_culture_lr[2],
            },
            "majority_age_x_culture": {
                "lr_stat": maj_interaction_lr[0],
                "df": maj_interaction_lr[1],
                "p_value": maj_interaction_lr[2],
            },
        },
        "descriptives": {
            "social_by_culture": social_by_culture.to_dict(orient="records"),
            "majority_by_culture": majority_by_culture.to_dict(orient="records"),
            "social_by_age_group": social_by_age.to_dict(orient="records"),
            "majority_by_age_group": majority_by_age.to_dict(orient="records"),
            "majority_bias_by_culture": maj_bias_by_culture.to_dict(orient="records"),
            "majority_bias_by_age_group": maj_bias_by_age.to_dict(orient="records"),
        },
    }

    # Write a machine-readable summary for inspection.
    Path("analysis_results.json").write_text(json.dumps(results, indent=2))

    # Also print a short human-readable summary to stdout.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

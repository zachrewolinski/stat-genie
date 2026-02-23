import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def fit_logit(formula: str, data: pd.DataFrame):
    """Fit a logistic regression model and return it (or None on failure)."""
    try:
        model = smf.logit(formula, data=data).fit(disp=False, maxiter=200)
        return model
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to fit model '{formula}': {exc}")
        return None


def lr_test(full, reduced):
    """Likelihood-ratio test comparing two nested models."""
    if full is None or reduced is None:
        return np.nan
    lr_stat = 2 * (full.llf - reduced.llf)
    df_diff = full.df_model - reduced.df_model
    if df_diff <= 0:
        return np.nan
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return float(p_value)


def main():
    # Load data
    df = pd.read_csv("boxes.csv")

    # Construct key derived variables
    df["social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    df_social = df[df["social"] == 1].copy()

    # Define coarse age groups to allow for non-linear developmental patterns
    def age_to_group(a: int) -> str:
        if a <= 6:
            return "4-6"
        if a <= 8:
            return "7-8"
        if a <= 10:
            return "9-10"
        if a <= 12:
            return "11-12"
        return "13-14"

    df["age_group"] = df["age"].apply(age_to_group)
    df_social["age_group"] = df_social["age"].apply(age_to_group)

    # ---------- Reliance on social information ----------
    social_null = fit_logit("social ~ 1", df)
    social_age = fit_logit("social ~ age", df)
    social_age_culture = fit_logit("social ~ age + C(culture)", df)

    p_age_social = lr_test(social_age, social_null)
    p_culture_social = lr_test(social_age_culture, social_age)

    coef_age_social = (
        float(social_age_culture.params.get("age")) if social_age_culture is not None else np.nan
    )

    # Predicted probabilities of using social info at youngest vs oldest ages
    p_social_age4 = np.nan
    p_social_age14 = np.nan
    if social_age_culture is not None:
        # Use culture=1 as reference for illustration
        base_row = {
            "age": [4, 14],
            "culture": [1, 1],
        }
        pred = social_age_culture.predict(pd.DataFrame(base_row))
        p_social_age4, p_social_age14 = float(pred.iloc[0]), float(pred.iloc[1])

    # Variation by culture and age-group (descriptive and chi-square tests)
    social_by_culture = df.groupby("culture")["social"].mean().to_dict()
    social_by_age_group = df.groupby("age_group")["social"].mean().to_dict()

    # Chi-square tests
    chi2_age_group_social = stats.chi2_contingency(
        pd.crosstab(df["age_group"], df["social"])
    )
    chi2_culture_social = stats.chi2_contingency(pd.crosstab(df["culture"], df["social"]))

    # ---------- Majority preference among social choosers ----------
    majority_null = fit_logit("majority_choice ~ 1", df_social)
    majority_age = fit_logit("majority_choice ~ age", df_social)
    majority_age_culture = fit_logit("majority_choice ~ age + C(culture)", df_social)

    p_age_majority = lr_test(majority_age, majority_null)
    p_culture_majority = lr_test(majority_age_culture, majority_age)

    coef_age_majority = (
        float(majority_age_culture.params.get("age")) if majority_age_culture is not None else np.nan
    )

    p_majority_age4 = np.nan
    p_majority_age14 = np.nan
    if majority_age_culture is not None:
        base_row = {
            "age": [4, 14],
            "culture": [1, 1],
        }
        pred = majority_age_culture.predict(pd.DataFrame(base_row))
        p_majority_age4, p_majority_age14 = float(pred.iloc[0]), float(pred.iloc[1])

    majority_rate = df_social["majority_choice"].mean()
    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean().to_dict()
    majority_by_age_group = (
        df_social.groupby("age_group")["majority_choice"].mean().to_dict()
    )

    chi2_age_group_majority = stats.chi2_contingency(
        pd.crosstab(df_social["age_group"], df_social["majority_choice"])
    )
    chi2_culture_majority = stats.chi2_contingency(
        pd.crosstab(df_social["culture"], df_social["majority_choice"])
    )

    # ---------- Package results ----------
    results = {
        "n": int(df.shape[0]),
        "social": {
            "p_age": p_age_social,
            "p_culture": p_culture_social,
            "coef_age": coef_age_social,
            "p_age4": p_social_age4,
            "p_age14": p_social_age14,
            "by_culture": social_by_culture,
            "by_age_group": social_by_age_group,
            "chi2_age_group": {
                "chi2": float(chi2_age_group_social[0]),
                "dof": int(chi2_age_group_social[2]),
                "p": float(chi2_age_group_social[1]),
            },
            "chi2_culture": {
                "chi2": float(chi2_culture_social[0]),
                "dof": int(chi2_culture_social[2]),
                "p": float(chi2_culture_social[1]),
            },
        },
        "majority": {
            "overall_rate": majority_rate,
            "p_age": p_age_majority,
            "p_culture": p_culture_majority,
            "coef_age": coef_age_majority,
            "p_age4": p_majority_age4,
            "p_age14": p_majority_age14,
            "by_culture": majority_by_culture,
            "by_age_group": majority_by_age_group,
            "chi2_age_group": {
                "chi2": float(chi2_age_group_majority[0]),
                "dof": int(chi2_age_group_majority[2]),
                "p": float(chi2_age_group_majority[1]),
            },
            "chi2_culture": {
                "chi2": float(chi2_culture_majority[0]),
                "dof": int(chi2_culture_majority[2]),
                "p": float(chi2_culture_majority[1]),
            },
        },
    }

    # Save numeric/statistical results to help manual interpretation if desired.
    Path("analysis_results.json").write_text(json.dumps(results, indent=2))

    # Also print a terse summary to stdout for interactive inspection.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

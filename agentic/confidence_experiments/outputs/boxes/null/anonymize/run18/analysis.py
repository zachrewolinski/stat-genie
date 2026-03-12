import json
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def fit_logit(formula: str, data: pd.DataFrame):
    """Fit a logistic regression model and return the fitted object."""
    model = smf.logit(formula=formula, data=data)
    result = model.fit(disp=False)
    return result


def lr_test(full_model, reduced_model) -> float:
    """Likelihood-ratio test p-value comparing full vs reduced logistic models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return float(p_value)


def site_pred_range(model, data: pd.DataFrame, response_col: str, age_value: float) -> float:
    """
    Compute the range of predicted probabilities across sites at a fixed age,
    holding gender and majority_first at their modal values.
    """
    mode_gender = data["gender"].mode().iloc[0]
    mode_majority_first = data["majority_first"].mode().iloc[0]

    sites = sorted(data["site"].unique())
    preds = []
    for s in sites:
        row = {
            "age": age_value,
            "site": s,
            "gender": mode_gender,
            "majority_first": mode_majority_first,
        }
        pred = float(model.predict(pd.DataFrame([row]))[0])
        preds.append(pred)

    return float(np.max(preds) - np.min(preds))


def age_effect(model, data: pd.DataFrame, age_low: float, age_high: float) -> float:
    """
    Compute change in predicted probability between two ages,
    holding site, gender, and majority_first at their modal values.
    """
    mode_site = data["site"].mode().iloc[0]
    mode_gender = data["gender"].mode().iloc[0]
    mode_majority_first = data["majority_first"].mode().iloc[0]

    def _pred(age_val: float) -> float:
        row = {
            "age": age_val,
            "site": mode_site,
            "gender": mode_gender,
            "majority_first": mode_majority_first,
        }
        return float(model.predict(pd.DataFrame([row]))[0])

    return float(_pred(age_high) - _pred(age_low))


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Basic sanity check on coding
    # choice: 1 = undemonstrated, 2 = majority, 3 = minority
    # gender: 1 = girl, 2 = boy
    # majority_first: 0/1
    # site: 1-8

    # Derived variables
    df["social_choice"] = (df["choice"] != 1).astype(int)

    df_social = df[df["social_choice"] == 1].copy()
    df_social["majority_vs_minority"] = (df_social["choice"] == 2).astype(int)

    # Descriptive statistics
    overall_social_rate = float(df["social_choice"].mean())
    overall_majority_among_social = float(df_social["majority_vs_minority"].mean())

    # Model 1: Reliance on social information (any demonstrated option vs undemonstrated)
    formula_social_full = "social_choice ~ age + C(site) + C(gender) + majority_first"
    formula_social_reduced = "social_choice ~ age + C(gender) + majority_first"

    model_social_full = fit_logit(formula_social_full, df)
    model_social_reduced = fit_logit(formula_social_reduced, df)

    social_age_p = float(model_social_full.pvalues["age"])
    social_site_p = lr_test(model_social_full, model_social_reduced)

    social_age_effect = age_effect(model_social_full, df, age_low=5.0, age_high=12.0)
    social_site_range = site_pred_range(
        model_social_full, df, response_col="social_choice", age_value=8.0
    )

    # Model 2: Majority preference among children who used social information
    formula_majority_full = (
        "majority_vs_minority ~ age + C(site) + C(gender) + majority_first"
    )
    formula_majority_reduced = "majority_vs_minority ~ age + C(gender) + majority_first"

    model_majority_full = fit_logit(formula_majority_full, df_social)
    model_majority_reduced = fit_logit(formula_majority_reduced, df_social)

    majority_age_p = float(model_majority_full.pvalues["age"])
    majority_site_p = lr_test(model_majority_full, model_majority_reduced)

    majority_age_effect = age_effect(
        model_majority_full, df_social, age_low=5.0, age_high=12.0
    )
    majority_site_range = site_pred_range(
        model_majority_full, df_social, response_col="majority_vs_minority", age_value=8.0
    )

    result: Dict[str, float] = {
        "n": float(len(df)),
        "social_choice_rate": overall_social_rate,
        "majority_among_social_rate": overall_majority_among_social,
        "social_age_p": social_age_p,
        "social_site_p": social_site_p,
        "social_age_effect_5_to_12": social_age_effect,
        "social_site_range_at_age8": social_site_range,
        "majority_age_p": majority_age_p,
        "majority_site_p": majority_site_p,
        "majority_age_effect_5_to_12": majority_age_effect,
        "majority_site_range_at_age8": majority_site_range,
    }

    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()


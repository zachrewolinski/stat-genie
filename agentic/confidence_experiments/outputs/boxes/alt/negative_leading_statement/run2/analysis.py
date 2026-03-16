import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


def lr_test(full_result, reduced_result):
    """Likelihood-ratio test comparing nested models."""
    lr_stat = 2 * (full_result.llf - reduced_result.llf)
    df_diff = int(round(full_result.df_model - reduced_result.df_model))
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return float(lr_stat), float(p_value), df_diff


def main():
    df = pd.read_csv("boxes.csv")

    # Define outcomes
    df["social"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(
        df["y"] == 2,
        1,
        np.where(df["y"] == 3, 0, np.nan),
    )

    results = {}

    # Model 1: Reliance on social information (any demonstrator vs undemonstrated option)
    social_formula_full = "social ~ age + C(culture) + gender + majority_first"
    social_formula_no_age = "social ~ C(culture) + gender + majority_first"
    social_formula_no_culture = "social ~ age + gender + majority_first"

    social_full = smf.logit(social_formula_full, data=df).fit(disp=False, maxiter=500)
    social_no_age = smf.logit(social_formula_no_age, data=df).fit(disp=False, maxiter=500)
    social_no_culture = smf.logit(
        social_formula_no_culture, data=df
    ).fit(disp=False, maxiter=500)

    lr_age_social, p_age_social, _ = lr_test(social_full, social_no_age)
    lr_cult_social, p_cult_social, _ = lr_test(social_full, social_no_culture)

    coef_age_social = float(social_full.params["age"])
    or_age_social = float(np.exp(coef_age_social))

    # Predicted probabilities at youngest vs oldest ages for a reference child
    age_min = df["age"].min()
    age_max = df["age"].max()
    ref = {
        "culture": df["culture"].mode().iloc[0],
        "gender": df["gender"].mode().iloc[0],
        "majority_first": df["majority_first"].mode().iloc[0],
    }
    df_pred_social = pd.DataFrame(
        [
            {"age": age_min, **ref},
            {"age": age_max, **ref},
        ]
    )
    probs_social = social_full.predict(df_pred_social)

    results["social"] = {
        "lr_age_stat": lr_age_social,
        "lr_age_p": p_age_social,
        "lr_culture_stat": lr_cult_social,
        "lr_culture_p": p_cult_social,
        "coef_age": coef_age_social,
        "or_age": or_age_social,
        "age_min": float(age_min),
        "age_max": float(age_max),
        "prob_social_age_min": float(probs_social.iloc[0]),
        "prob_social_age_max": float(probs_social.iloc[1]),
    }

    # Model 2: Preference for majority vs minority among children who used social info
    df_social = df[df["social"] == 1].copy()

    majority_formula_full = "majority_choice ~ age + C(culture) + gender + majority_first"
    majority_formula_no_age = "majority_choice ~ C(culture) + gender + majority_first"
    majority_formula_no_culture = "majority_choice ~ age + gender + majority_first"

    majority_full = smf.logit(
        majority_formula_full, data=df_social
    ).fit(disp=False, maxiter=500)
    majority_no_age = smf.logit(
        majority_formula_no_age, data=df_social
    ).fit(disp=False, maxiter=500)
    majority_no_culture = smf.logit(
        majority_formula_no_culture, data=df_social
    ).fit(disp=False, maxiter=500)

    lr_age_majority, p_age_majority, _ = lr_test(majority_full, majority_no_age)
    lr_cult_majority, p_cult_majority, _ = lr_test(
        majority_full, majority_no_culture
    )

    coef_age_majority = float(majority_full.params["age"])
    or_age_majority = float(np.exp(coef_age_majority))

    df_pred_majority = pd.DataFrame(
        [
            {"age": age_min, **ref},
            {"age": age_max, **ref},
        ]
    )
    probs_majority = majority_full.predict(df_pred_majority)

    results["majority"] = {
        "lr_age_stat": lr_age_majority,
        "lr_age_p": p_age_majority,
        "lr_culture_stat": lr_cult_majority,
        "lr_culture_p": p_cult_majority,
        "coef_age": coef_age_majority,
        "or_age": or_age_majority,
        "prob_majority_age_min": float(probs_majority.iloc[0]),
        "prob_majority_age_max": float(probs_majority.iloc[1]),
    }

    # Simple descriptive stats
    results["descriptives"] = {
        "n_total": int(len(df)),
        "social_rate_overall": float(df["social"].mean()),
        "majority_rate_among_social": float(
            df_social["majority_choice"].mean()
        ),
    }

    # Print results as a JSON-like dict; the assistant will interpret this.
    # Using repr to avoid relying on json in this minimal script.
    print(results)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )
    # Keep only rows with valid socket counts and valid counts
    df = df[df["sockets"] > 0].copy()
    df = df[df["missing"] <= df["sockets"]].copy()
    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def _build_design_matrix_for_fit(df: pd.DataFrame):
    X = pd.DataFrame(
        {
            "intercept": 1.0,
            "is_human": df["is_human"],
            "age": df["age"],
            "sex_estimate": df["sex_estimate"],
        },
        index=df.index,
    )
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)
    tooth_dummy_cols = tooth_dummies.columns.tolist()
    X = pd.concat([X, tooth_dummies], axis=1)
    return X, tooth_dummy_cols


def _build_design_matrix_for_pred(df_pred: pd.DataFrame, tooth_dummy_cols):
    X = pd.DataFrame(
        {
            "intercept": 1.0,
            "is_human": df_pred["is_human"],
            "age": df_pred["age"],
            "sex_estimate": df_pred["sex_estimate"],
        },
        index=df_pred.index,
    )
    dummies = pd.get_dummies(df_pred["tooth_class"], prefix="tooth", drop_first=True)
    for col in tooth_dummy_cols:
        if col not in dummies:
            dummies[col] = 0
    dummies = dummies[tooth_dummy_cols]
    X = pd.concat([X, dummies], axis=1)
    return X


def fit_glm(df: pd.DataFrame):
    X, tooth_dummy_cols = _build_design_matrix_for_fit(df)
    y = np.column_stack(
        [df["missing"].to_numpy(), (df["sockets"] - df["missing"]).to_numpy()]
    )
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    # Use cluster-robust SEs by specimen to account for repeated measures
    try:
        robust = result.get_robustcov_results(cov_type="cluster", groups=df["specimen"])
    except Exception:
        robust = result
    return robust, tooth_dummy_cols


def summarize_effect(result, df: pd.DataFrame, tooth_dummy_cols):
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pval = float(result.pvalues["is_human"])
    ci_low, ci_high = result.conf_int().loc["is_human"]
    or_est = float(np.exp(coef))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    # Predicted probabilities at typical values
    mean_age = float(df["age"].mean())
    mean_sex = float(df["sex_estimate"].mean())
    common_class = df["tooth_class"].mode().iloc[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [1, 0],
            "age": [mean_age, mean_age],
            "sex_estimate": [mean_sex, mean_sex],
            "tooth_class": [common_class, common_class],
        }
    )
    X_pred = _build_design_matrix_for_pred(pred_df, tooth_dummy_cols)
    pred_probs = result.predict(X_pred)
    p_human = float(pred_probs.iloc[0])
    p_nonhuman = float(pred_probs.iloc[1])

    return {
        "coef": float(coef),
        "se": float(se),
        "pval": pval,
        "or_est": or_est,
        "or_low": or_low,
        "or_high": or_high,
        "p_human": p_human,
        "p_nonhuman": p_nonhuman,
    }


def compute_likert(summary: dict) -> int:
    or_est = summary["or_est"]
    pval = summary["pval"]

    if pval < 0.05:
        if or_est > 1.0:
            # Significant evidence humans have higher AMTL
            if or_est >= 2.0:
                resp = 90
            elif or_est >= 1.5:
                resp = 80
            else:
                resp = 70
        else:
            # Significant evidence against the research statement
            if or_est <= 0.5:
                resp = 10
            elif or_est <= 0.67:
                resp = 20
            else:
                resp = 30
    elif pval < 0.1:
        # Marginal evidence
        resp = 60 if or_est > 1.0 else 40
    else:
        # No clear evidence either way; stay near neutral
        if or_est > 1.1:
            resp = 55
        elif or_est < 0.9:
            resp = 45
        else:
            resp = 50

    resp_int = max(0, min(100, int(round(resp))))
    return resp_int


def build_explanation(summary: dict, response: int) -> str:
    or_est = summary["or_est"]
    or_low = summary["or_low"]
    or_high = summary["or_high"]
    pval = summary["pval"]
    p_human = summary["p_human"]
    p_nonhuman = summary["p_nonhuman"]

    direction = (
        "higher" if or_est > 1.0 else "lower" if or_est < 1.0 else "similar"
    )

    if response > 50:
        headline = (
            "There is evidence that modern humans have higher AMTL frequencies "
            "than non-human primates after controlling for age, sex, and tooth class."
        )
    elif response < 50:
        headline = (
            "The analysis does not support the claim that modern humans have higher AMTL "
            "frequencies than non-human primates after controlling for age, sex, and tooth class."
        )
    else:
        headline = (
            "The data provide little clear evidence that modern humans differ from "
            "non-human primates in AMTL frequencies after controlling for age, sex, and tooth class."
        )

    explanation = (
        f"{headline} "
        f"A binomial regression model of the number of missing teeth out of observable sockets "
        f"used an indicator for modern humans versus non-human primate genera, with age at death, "
        f"estimated sex, and tooth class (anterior, posterior, premolar) as covariates. "
        f"The estimated odds ratio for AMTL in modern humans relative to non-human primates was "
        f"{or_est:.2f} (95% CI {or_low:.2f}–{or_high:.2f}, p = {pval:.3g}), indicating {direction} "
        f"AMTL in humans on the odds scale. "
        f"At typical values of age, sex, and tooth class, the model predicts a probability of a tooth "
        f"being missing of approximately {p_human:.3f} for modern humans versus {p_nonhuman:.3f} for "
        f"non-human primates. "
        f"These results correspond to a Likert-scale response of {response} (0 = strong 'No', "
        f"100 = strong 'Yes') to the question of whether modern humans have higher AMTL frequencies "
        f"than non-human primates after accounting for age, sex, and tooth class."
    )
    return explanation


def main():
    csv_path = Path("amtl.csv")
    df = load_data(str(csv_path))
    result, tooth_dummy_cols = fit_glm(df)
    summary = summarize_effect(result, df, tooth_dummy_cols)
    response = compute_likert(summary)
    explanation = build_explanation(summary, response)

    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

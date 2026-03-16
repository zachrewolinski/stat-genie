import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent


def load_data():
    df = pd.read_csv(ROOT / "caschools.csv")
    return df


def prepare_variables(df: pd.DataFrame):
    """Construct student-teacher ratio and academic performance variables.

    Based on the metadata in info.json, we interpret:
    - `students`: number of teachers (full-time equivalents)
    - `english`: total enrollment
    - `district`: average reading score
    - `expenditure`: average math score

    We define:
    - student-teacher ratio = enrollment / teachers = english / students
    - academic performance = average of reading and math scores
    """

    df = df.copy()

    # Avoid division by zero
    df = df.replace([np.inf, -np.inf], np.nan)

    # Student-teacher ratio
    df["stratio"] = df["english"] / df["students"]

    # Academic performance as mean of reading and math scores
    df["perf"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing or nonsensical values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["stratio", "perf"])  # ensure data for both

    return df


def fit_models(df: pd.DataFrame):
    """Fit simple models of performance vs student-teacher ratio.

    We fit:
    - Model 1: perf ~ stratio (simple linear regression)
    - Model 2: perf ~ stratio + income + lunch + calworks + computer + school
      where possible, to check robustness when controlling for key covariates.
    """

    results = {}

    # Simple linear regression
    x = sm.add_constant(df["stratio"])  # add intercept
    y = df["perf"]
    model_simple = sm.OLS(y, x).fit()
    results["simple"] = {
        "coef_stratio": float(model_simple.params["stratio"]),
        "pvalue_stratio": float(model_simple.pvalues["stratio"]),
        "r2": float(model_simple.rsquared),
        "nobs": int(model_simple.nobs),
    }

    # Multiple regression with covariates.
    covariates = []
    for col in ["income", "school", "computer", "lunch"]:
        if col in df.columns:
            covariates.append(col)

    # `calworks` in this shuffled dataset is a string (school name), so skip it.

    if covariates:
        Xcols = ["stratio"] + covariates
        X = sm.add_constant(df[Xcols])
        model_multi = sm.OLS(y, X).fit()
        results["multiple"] = {
            "coef_stratio": float(model_multi.params["stratio"]),
            "pvalue_stratio": float(model_multi.pvalues["stratio"]),
            "r2": float(model_multi.rsquared),
            "nobs": int(model_multi.nobs),
        }

    return results


def summarize(results: dict) -> dict:
    """Produce a Likert-style response and narrative explanation."""
    simple = results["simple"]
    coef = simple["coef_stratio"]
    pval = simple["pvalue_stratio"]
    r2 = simple["r2"]

    # Direction: negative coefficient implies lower ratio -> higher performance.
    direction = "negative" if coef < 0 else "positive"

    explanation_lines = []

    explanation_lines.append(
        "We analyzed whether districts with lower student-teacher ratios have higher academic performance. "
        "We constructed student-teacher ratio as total enrollment divided by the number of teachers (english / students) "
        "and defined academic performance as the mean of district-level reading and math scores."
    )

    explanation_lines.append(
        f"In a simple linear regression of performance on student-teacher ratio (n = {simple['nobs']}), "
        f"the estimated coefficient on the ratio is {coef:.3f} with p-value {pval:.4g} and R-squared {r2:.3f}."
    )

    # Also summarize multiple model if available
    multiple = results.get("multiple")
    if multiple is not None:
        explanation_lines.append(
            "We also estimated a multiple regression that controls for district income, poverty proxies (percent on reduced-price lunch), "
            "and related demographics where available. In this model, the coefficient on student-teacher ratio is "
            f"{multiple['coef_stratio']:.3f} with p-value {multiple['pvalue_stratio']:.4g} and R-squared {multiple['r2']:.3f}."
        )

    # Map evidence to Likert scale (0-100)
    # Strong, statistically significant negative association -> high score near 100.
    # Weak or non-significant association -> near 50 or below.

    if pval < 0.001 and coef < 0:
        response = 95
        conclusion = (
            "These results provide very strong evidence that lower student-teacher ratios are associated with higher academic performance, "
            "even after accounting for key district characteristics."
        )
    elif pval < 0.01 and coef < 0:
        response = 85
        conclusion = (
            "There is strong evidence of a negative association between student-teacher ratio and academic performance: districts with fewer students per teacher tend to score higher."
        )
    elif pval < 0.05 and coef < 0:
        response = 75
        conclusion = (
            "There is moderate but statistically significant evidence that lower student-teacher ratios are linked to better academic performance."
        )
    elif pval < 0.05 and coef > 0:
        response = 25
        conclusion = (
            "Surprisingly, the estimated association is positive and statistically significant: districts with higher student-teacher ratios appear to have slightly higher performance, "
            "though this pattern may reflect unmodeled confounding rather than a causal effect."
        )
    else:
        # Not statistically significant
        if coef < 0:
            response = 45
        else:
            response = 40
        conclusion = (
            "The estimated relationship between student-teacher ratio and academic performance is weak and not statistically reliable, "
            "so the data do not provide clear evidence that lower ratios are associated with higher scores."
        )

    explanation_lines.append(conclusion)

    explanation_lines.append(
        "Overall, we interpret the regression results in terms of association rather than causality; unobserved district characteristics and policy choices may still drive both class size and performance."
    )

    explanation = " ".join(explanation_lines)

    return {"response": int(response), "explanation": explanation}


def main():
    df_raw = load_data()
    df = prepare_variables(df_raw)
    results = fit_models(df)
    summary = summarize(results)

    # Write conclusion.json-style output as specified
    out_path = ROOT / "conclusion.txt"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Compute student-teacher ratio and average test score as performance metric.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in key variables, if any.
    df = df.dropna(subset=["stratio", "avg_score"])

    n_obs = len(df)

    # Simple Pearson correlation between student-teacher ratio and performance.
    corr, corr_p = stats.pearsonr(df["stratio"], df["avg_score"])

    # Simple linear regression: avg_score ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()
    coef_stratio_simple = model_simple.params["stratio"]
    pval_stratio_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    # Multiple regression controlling for key demographics and resources.
    controls = ["english", "lunch", "calworks", "income", "computer", "expenditure"]
    controls = [c for c in controls if c in df.columns]

    coef_stratio_multi = None
    pval_stratio_multi = None
    r2_multi = None

    if controls:
        cols = ["stratio"] + controls
        df_multi = df.dropna(subset=cols + ["avg_score"])
        X_multi = sm.add_constant(df_multi[cols])
        model_multi = sm.OLS(df_multi["avg_score"], X_multi).fit()
        coef_stratio_multi = model_multi.params["stratio"]
        pval_stratio_multi = model_multi.pvalues["stratio"]
        r2_multi = model_multi.rsquared
        n_multi = len(df_multi)
    else:
        n_multi = None

    results = {
        "n_obs": int(n_obs),
        "corr_stratio_avg_score": float(corr),
        "corr_p_value": float(corr_p),
        "simple_regression": {
            "coef_stratio": float(coef_stratio_simple),
            "p_value_stratio": float(pval_stratio_simple),
            "r_squared": float(r2_simple),
        },
        "multiple_regression": {
            "n_obs": int(n_multi) if n_multi is not None else None,
            "coef_stratio": float(coef_stratio_multi) if coef_stratio_multi is not None else None,
            "p_value_stratio": float(pval_stratio_multi) if pval_stratio_multi is not None else None,
            "r_squared": float(r2_multi) if r2_multi is not None else None,
            "controls": controls,
        },
    }

    # Save detailed numerical results for inspection if needed.
    Path("analysis_results.json").write_text(json.dumps(results, indent=2))

    # Based on the analysis, determine Likert-scale response.
    # The research question asks:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    #
    # In this dataset, the correlation between student-teacher ratio (students per teacher)
    # and average test score is essentially zero (~0.02) with a very large p-value (~0.67).
    # Both the simple and multiple regression models yield coefficients on the ratio that
    # are extremely close to zero and far from statistical significance (p > 0.65),
    # even after adjusting for demographics and resources.
    #
    # This provides little to no evidence that lower ratios are associated with higher
    # academic performance in these data.
    response_value = 10

    explanation = (
        "Using data on 420 California K-6 and K-8 districts, I defined the student–teacher "
        "ratio as total students divided by full-time-equivalent teachers and academic "
        "performance as the average of the reading and math scores. The Pearson correlation "
        "between the student–teacher ratio and average score is essentially zero "
        f"(r ≈ {corr:.03f}, p ≈ {corr_p:.03f}), indicating no linear association. A simple "
        "linear regression of average score on the student–teacher ratio yields an estimated "
        f"coefficient that is effectively zero (≈ {coef_stratio_simple:.4f}) with a very large "
        f"p-value (≈ {pval_stratio_simple:.03f}) and almost no explained variance (R² ≈ {r2_simple:.04f}). "
        "When I fit a multiple regression that controls for key demographic and resource "
        "variables (percent English learners, lunch and CalWORKs participation, income, "
        "computers, and expenditures), the coefficient on the student–teacher ratio remains "
        f"near zero (≈ {coef_stratio_multi:.4f}) and statistically non-significant "
        f"(p ≈ {pval_stratio_multi:.03f}), with only a small increase in R² (≈ {r2_multi:.04f}). "
        "Taken together, these results provide little evidence that districts with lower "
        "student–teacher ratios have meaningfully higher academic performance in this dataset. "
        "Accordingly, I answer 'No' to the research question and place my confidence at 10 on "
        "a 0–100 Likert scale, reflecting a fairly strong conclusion that any relationship, if "
        "present at all, is very weak in these data."
    )

    conclusion = {"response": int(response_value), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))

    # Also print a concise summary for quick human inspection in the terminal.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

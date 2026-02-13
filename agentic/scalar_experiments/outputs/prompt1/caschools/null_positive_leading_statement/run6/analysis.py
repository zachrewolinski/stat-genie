import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Student–teacher ratio: number of students per full-time-equivalent teacher.
    df["stratio"] = df["students"] / df["teachers"]
    # Overall academic performance: average of reading and math scores.
    df["score"] = df[["read", "math"]].mean(axis=1)
    return df


def correlation_analysis(df: pd.DataFrame) -> dict:
    corr, pval = stats.pearsonr(df["stratio"], df["score"])
    return {
        "corr_stratio_score": float(corr),
        "pval_stratio_score": float(pval),
    }


def regression_analysis(df: pd.DataFrame) -> dict:
    # Include key demographic and resource controls to check robustness of the association.
    covariates = ["stratio", "income", "english", "calworks", "lunch", "expenditure"]
    X = df[covariates].copy()
    X = sm.add_constant(X)
    y = df["score"]

    model = sm.OLS(y, X).fit()
    coef = model.params["stratio"]
    pval = model.pvalues["stratio"]

    return {
        "coef_stratio": float(coef),
        "pval_stratio": float(pval),
        "r_squared": float(model.rsquared),
    }


def main() -> None:
    data_path = Path("caschools.csv")
    df = load_data(data_path)

    corr_results = correlation_analysis(df)
    reg_results = regression_analysis(df)

    # Decide whether there is evidence that lower student–teacher ratios
    # are associated with higher academic performance.
    corr = corr_results["corr_stratio_score"]
    corr_pval = corr_results["pval_stratio_score"]
    coef = reg_results["coef_stratio"]
    coef_pval = reg_results["pval_stratio"]

    # Because stratio is students per teacher, a *negative* association with score
    # means that lower ratios (smaller classes) are associated with higher scores.
    alpha = 0.05

    # Evidence criteria: statistically significant negative correlation and regression coefficient.
    has_negative_corr = (corr < 0) and (corr_pval < alpha)
    has_negative_coef = (coef < 0) and (coef_pval < alpha)

    if has_negative_corr and has_negative_coef:
        response = "Yes"
    else:
        response = "No"

    explanation_parts = [
        "We analyzed data on 420 California K-6 and K-8 districts, "
        "constructing a student–teacher ratio (students per teacher) and an overall academic "
        "performance score (average of reading and math Stanford 9 test scores).",
        f"The simple Pearson correlation between student–teacher ratio and academic performance "
        f"was {corr:.3f} (p-value = {corr_pval:.3g}).",
        f"We then estimated an OLS regression of academic performance on student–teacher ratio, "
        f"controlling for district average income, percent English learners, percent of students "
        f"on CalWorks, percent qualifying for reduced-price lunch, and per-student expenditure. "
        f"The coefficient on student–teacher ratio was {coef:.3f} (p-value = {coef_pval:.3g}), "
        f"with model R-squared of {reg_results['r_squared']:.3f}.",
    ]

    if response == "Yes":
        explanation_parts.append(
            "In both the correlation and regression analyses, higher student–teacher ratios "
            "(larger classes) were significantly associated with lower test scores, implying that "
            "lower student–teacher ratios are associated with higher academic performance in this dataset."
        )
    else:
        explanation_parts.append(
            "The estimated association between student–teacher ratio and academic performance was not "
            "consistently negative and statistically significant across both correlation and regression "
            "analyses, so this dataset does not provide strong evidence that lower student–teacher ratios "
            "are associated with higher academic performance."
        )

    conclusion = {
        "response": response,
        "explanation": " ".join(explanation_parts),
    }

    # Write conclusion.txt exactly as required.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


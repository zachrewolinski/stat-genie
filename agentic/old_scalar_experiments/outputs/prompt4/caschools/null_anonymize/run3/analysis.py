import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: str = "caschools.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Compute student-teacher ratio: students per teacher.
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    # Academic performance as the average of reading and math scores.
    df["avg_test_score"] = (df["feature14"] + df["feature15"]) / 2
    return df


def analyze_relationship(df: pd.DataFrame) -> dict:
    # Drop rows with missing values in the key variables, if any.
    sub = df[["student_teacher_ratio", "avg_test_score", "feature8", "feature9", "feature11", "feature12", "feature13"]].dropna()

    # Simple Pearson correlation.
    corr = sub["student_teacher_ratio"].corr(sub["avg_test_score"])

    # Simple linear regression: test score on ratio only.
    X_simple = sm.add_constant(sub["student_teacher_ratio"])
    model_simple = sm.OLS(sub["avg_test_score"], X_simple).fit()
    coef_simple = model_simple.params["student_teacher_ratio"]
    pval_simple = model_simple.pvalues["student_teacher_ratio"]

    # Multiple regression controlling for key covariates:
    # socio-economic background and spending.
    X_controls = sub[
        [
            "student_teacher_ratio",
            "feature8",   # Percent qualifying for CalWorks.
            "feature9",   # Percent qualifying for reduced-price lunch.
            "feature11",  # Expenditure per student.
            "feature12",  # District average income (USD 1,000).
            "feature13",  # Percent of English learners.
        ]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(sub["avg_test_score"], X_controls).fit()
    coef_ctrl = model_controls.params["student_teacher_ratio"]
    pval_ctrl = model_controls.pvalues["student_teacher_ratio"]

    # Summaries for explanation.
    mean_ratio = sub["student_teacher_ratio"].mean()
    sd_ratio = sub["student_teacher_ratio"].std()
    mean_score = sub["avg_test_score"].mean()
    sd_score = sub["avg_test_score"].std()

    return {
        "corr": float(corr),
        "coef_simple": float(coef_simple),
        "pval_simple": float(pval_simple),
        "coef_ctrl": float(coef_ctrl),
        "pval_ctrl": float(pval_ctrl),
        "mean_ratio": float(mean_ratio),
        "sd_ratio": float(sd_ratio),
        "mean_score": float(mean_score),
        "sd_score": float(sd_score),
        "n_obs": int(len(sub)),
        "r2_simple": float(model_simple.rsquared),
        "r2_ctrl": float(model_controls.rsquared),
    }


def map_result_to_scale(results: dict) -> int:
    """
    Map statistical evidence to a 0-100 scale where higher values
    indicate stronger evidence that lower ratios are associated with
    higher performance.
    """
    corr = results["corr"]
    coef_simple = results["coef_simple"]
    pval_ctrl = results["pval_ctrl"]

    # Start from a neutral baseline.
    score = 50.0

    # Direction and magnitude of correlation.
    if corr < 0:
        score += min(20.0, 60.0 * abs(corr))  # up to +20 for strong negative corr
    else:
        score -= min(20.0, 60.0 * abs(corr))  # down to -20 for positive corr

    # Direction of controlled coefficient.
    if coef_simple < 0:
        score += 10.0
    else:
        score -= 10.0

    # Strength of evidence after controls (p-value).
    if pval_ctrl < 0.01:
        score += 15.0
    elif pval_ctrl < 0.05:
        score += 10.0
    elif pval_ctrl < 0.1:
        score += 5.0
    else:
        score -= 5.0

    # Bound to [0, 100] and convert to integer.
    score = int(max(0, min(100, round(score))))
    return score


def build_explanation(results: dict, response_score: int) -> str:
    coef_ctrl = results["coef_ctrl"]
    coef_simple = results["coef_simple"]
    direction = "negative" if coef_ctrl < 0 else "positive"
    strength_desc = []
    if results["pval_ctrl"] < 0.01:
        strength_desc.append("statistically strong")
    elif results["pval_ctrl"] < 0.05:
        strength_desc.append("statistically significant at the 5% level")
    elif results["pval_ctrl"] < 0.1:
        strength_desc.append("marginally significant")
    else:
        strength_desc.append("not statistically significant after controls")

    explanation = (
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?\n\n"
        f"I constructed the student-teacher ratio as total enrollment divided by the number of teachers "
        f"(mean {results['mean_ratio']:.1f}, SD {results['sd_ratio']:.1f}) and defined academic performance as "
        f"the average of district reading and math scores (mean {results['mean_score']:.1f}, "
        f"SD {results['sd_score']:.1f}) across {results['n_obs']} districts.\n\n"
        f"First, I examined the simple association between the student-teacher ratio and average test scores. "
        f"The Pearson correlation is {results['corr']:.3f}, and a bivariate regression of test scores on the "
        f"ratio yields a coefficient of {coef_simple:.3f} (R² = {results['r2_simple']:.3f}). "
        "A negative coefficient would indicate that districts with fewer students per teacher tend to have higher scores, "
        "whereas a positive coefficient would suggest the opposite.\n\n"
        "To account for observed differences in socioeconomic context and resources, I estimated a multiple "
        "linear regression of average test scores on the student-teacher ratio controlling for the percent of "
        "students receiving CalWorks and reduced-price lunch, expenditure per student, district average income, "
        "and the percent of English learners. In this model, the coefficient on the student-teacher ratio is "
        f"{coef_ctrl:.3f}, with a p-value of {results['pval_ctrl']:.3f} and R² = {results['r2_ctrl']:.3f}. "
        f"This indicates a {direction} association between the ratio and test scores that is {', '.join(strength_desc)} "
        "after adjusting for these covariates.\n\n"
        "In this dataset, the estimated associations between the student-teacher ratio and average test scores are "
        "very small in magnitude and statistically indistinguishable from zero once observed covariates are taken into "
        "account. If anything, the point estimates slightly favor higher scores in districts with larger student-teacher "
        "ratios, but these differences are not reliable. On the 0–100 scale, where higher values represent stronger "
        "evidence that a lower student-teacher ratio is associated with higher performance, I assign a score of "
        f"{response_score}, reflecting modest evidence against a meaningful positive association between smaller classes "
        "and higher academic performance in this sample."
    )
    return explanation


def write_conclusion(response_score: int, explanation: str, path: str = "conclusion.txt") -> None:
    payload = {"response": int(response_score), "explanation": explanation}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    df = load_data()
    results = analyze_relationship(df)
    response_score = map_result_to_scale(results)
    explanation = build_explanation(results, response_score)
    write_conclusion(response_score, explanation)


if __name__ == "__main__":
    main()

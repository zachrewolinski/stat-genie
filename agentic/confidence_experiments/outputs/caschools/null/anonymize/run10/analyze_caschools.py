import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def build_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze whether a lower student-teacher ratio is associated with higher academic performance.
    Returns a dict with keys:
      - response: int in [0, 100]
      - explanation: str
    """

    # Compute student-teacher ratio (students per teacher).
    df = df.copy()
    df["stu_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Composite academic performance: mean of reading and math scores.
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Keep relevant variables and drop missing / infinite values.
    cols = [
        "stu_teacher_ratio",
        "avg_score",
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature10",  # number of computers
        "feature11",  # expenditure per student
        "feature12",  # average income (1,000s)
        "feature13",  # % English learners
        "feature6",   # enrollment
    ]
    data = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    n_obs = int(data.shape[0])

    y = data["avg_score"]

    # Simple bivariate relationship.
    X_simple = sm.add_constant(data["stu_teacher_ratio"])
    model_simple = sm.OLS(y, X_simple).fit()

    # Multiple regression with key demographic and resource controls.
    X_controls = data[
        [
            "stu_teacher_ratio",
            "feature8",
            "feature9",
            "feature10",
            "feature11",
            "feature12",
            "feature13",
            "feature6",
        ]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(y, X_controls).fit()
    model_controls_robust = model_controls.get_robustcov_results(cov_type="HC3")

    # Core statistics for student-teacher ratio.
    coef_simple = float(model_simple.params["stu_teacher_ratio"])
    p_simple = float(model_simple.pvalues["stu_teacher_ratio"])

    # For the robust model, params and pvalues are numpy arrays, so index by position.
    exog_names = list(model_controls_robust.model.exog_names)
    ratio_idx = exog_names.index("stu_teacher_ratio")
    coef_controls = float(model_controls_robust.params[ratio_idx])
    p_controls = float(model_controls_robust.pvalues[ratio_idx])

    r2_simple = float(model_simple.rsquared)
    r2_controls = float(model_controls.rsquared)

    corr = float(data["stu_teacher_ratio"].corr(data["avg_score"]))

    # Build an evidence-based Likert score between 0 and 100.
    # Start neutral.
    score = 50.0

    # Direction: we expect a negative coefficient if lower ratios
    # (fewer students per teacher) are associated with higher scores.
    if coef_simple < 0 and coef_controls < 0:
        score += 15.0
    elif coef_simple < 0 or coef_controls < 0:
        score += 5.0
    else:
        # Coefficients not in expected direction.
        score -= 20.0

    # Statistical significance from both models.
    def p_to_bonus(p_val: float) -> float:
        if p_val < 0.01:
            return 15.0
        if p_val < 0.05:
            return 8.0
        if p_val < 0.1:
            return 4.0
        return -6.0

    score += p_to_bonus(p_simple)
    score += p_to_bonus(p_controls)

    # Effect size via correlation magnitude.
    score += 20.0 * min(abs(corr), 0.5)  # cap at |corr| = 0.5

    # Clamp to [0, 100] and convert to int.
    score = max(0.0, min(100.0, score))
    response_int = int(round(score))

    # Build human-readable explanation.
    def fmt(x: float, digits: int = 3) -> str:
        return f"{x:.{digits}f}"

    direction_text = "negative" if coef_controls < 0 else "positive"

    # High-level conclusion depends on sign, magnitude, and significance.
    strong_negative = (
        coef_simple < 0
        and coef_controls < 0
        and p_simple < 0.05
        and p_controls < 0.05
    )
    near_zero = (abs(corr) < 0.1) and (p_simple > 0.1) and (p_controls > 0.1)

    if strong_negative:
        conclusion_sentence = (
            "Because the student-teacher ratio shows a consistent, statistically significant "
            "negative association with test scores across both simple and controlled models, "
            "the evidence supports the claim that lower student-teacher ratios are associated "
            "with higher academic performance in this dataset. "
            "However, as these are observational data, the results describe association rather "
            "than definitive causation."
        )
    elif near_zero:
        conclusion_sentence = (
            "Because the estimated association between the student-teacher ratio and test scores "
            "is very small in magnitude and not statistically distinguishable from zero in either "
            "the simple or controlled models, the data do not provide strong evidence that lower "
            "student-teacher ratios are associated with higher academic performance in this dataset. "
            "The results are most consistent with little to no linear relationship at the district level."
        )
    else:
        conclusion_sentence = (
            "Overall, the estimates for the student-teacher ratio are not robustly different from zero "
            "across specifications, so while there are hints of the expected direction in some models, "
            "the data provide at best weak and inconclusive evidence that lower student-teacher ratios "
            "are associated with higher academic performance. Any such relationship, if present, is likely "
            "modest relative to other district characteristics."
        )

    explanation = (
        "Using data on 420 K-6 and K-8 school districts in California, "
        "I evaluated whether districts with lower student-teacher ratios tend to have "
        "higher academic performance (average of reading and math scores on the "
        "Stanford 9 test for 5th graders).\n\n"
        f"- After computing the student-teacher ratio as total enrollment divided by the number of teachers, "
        f"its simple correlation with the average test score is {fmt(corr)}.\n"
        f"- In a bivariate OLS regression of average score on the student-teacher ratio (n = {n_obs}), "
        f"the coefficient on the ratio is {fmt(coef_simple)} score points per additional student per teacher "
        f"(p = {fmt(p_simple)}, R² = {fmt(r2_simple)}).\n"
        "- I then estimated a multiple regression including controls for percent of students on CalWorks, "
        "percent eligible for reduced-price lunch, number of computers, expenditure per student, "
        "average district income, percent English learners, and total enrollment.\n"
        f"- In this controlled model, the coefficient on the student-teacher ratio remains {direction_text} "
        f"at {fmt(coef_controls)} with a p-value of {fmt(p_controls)} and R² = {fmt(r2_controls)} "
        "using heteroskedasticity-robust standard errors.\n\n"
        f"{conclusion_sentence}"
    )

    return {"response": response_int, "explanation": explanation}


def main() -> None:
    df = pd.read_csv("caschools.csv")
    result = build_analysis(df)

    # Write JSON conclusion to conclusion.txt with the required keys.
    Path("conclusion.txt").write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

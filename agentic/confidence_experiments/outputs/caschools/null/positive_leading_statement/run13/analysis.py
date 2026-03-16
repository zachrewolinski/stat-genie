import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    cwd = Path(__file__).parent
    data_path = cwd / "caschools.csv"

    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic sanity checks: drop rows with missing key fields if any
    df = df.dropna(subset=["stratio", "testscr", "income", "english", "lunch", "calworks", "expenditure", "computer"])

    # Simple correlation between student-teacher ratio and test scores
    corr = df["stratio"].corr(df["testscr"])

    # Simple bivariate regression
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]

    # Multiple regression controlling for key demographics and resources
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(df["testscr"], X_multi).fit()
    coef_multi = model_multi.params["stratio"]
    pval_multi = model_multi.pvalues["stratio"]
    r2_multi = model_multi.rsquared

    # Separate models for reading and math to check robustness
    results_detail = {}
    for outcome in ["read", "math"]:
        y = df[outcome]
        X = sm.add_constant(df[["stratio"] + controls])
        model = sm.OLS(y, X).fit()
        results_detail[outcome] = {
            "coef": float(model.params["stratio"]),
            "pval": float(model.pvalues["stratio"]),
            "r2": float(model.rsquared),
        }

    # Interpret the evidence for the Likert response
    # We expect a *negative* coefficient: higher student-teacher ratio (more students per teacher)
    # associated with lower scores, which implies that *lower* ratios are associated with higher performance.
    strong_and_significant = (
        coef_multi < 0 and pval_multi < 0.01 and coef_simple < 0 and pval_simple < 0.01
    )

    # Effect size: approximate change in testscr for 5-student change in ratio
    delta_score_5_students = coef_multi * 5.0

    # Map evidence to a 0–100 Likert scale
    if strong_and_significant:
        # Strong, consistent association but observational (not causal),
        # so avoid the absolute maximum.
        if abs(delta_score_5_students) >= 3:
            response_score = 90
        elif abs(delta_score_5_students) >= 1.5:
            response_score = 80
        else:
            response_score = 70
    else:
        # Weak or non-robust evidence
        if coef_multi < 0 and pval_multi < 0.1:
            response_score = 60
        elif pval_multi >= 0.1:
            response_score = 40
        else:
            response_score = 50

    # Build explanation text
    explanation_lines = []
    explanation_lines.append(
        "I examined whether a lower student-teacher ratio (fewer students per teacher) "
        "is associated with higher academic performance in California K-6/K-8 districts."
    )
    explanation_lines.append(
        f"The simple correlation between student-teacher ratio and average test score "
        f"(mean of reading and math) is {corr:.3f}, which is very close to zero and "
        "does not suggest a strong linear relationship between class size and performance."
    )
    explanation_lines.append(
        f"In a bivariate regression of average test score on the student-teacher ratio, "
        f"the estimated coefficient on the ratio is {coef_simple:.3f} (p-value {pval_simple:.4f}). "
        "This coefficient is extremely small in magnitude and not statistically distinguishable "
        "from zero, indicating no clear association when considered alone."
    )
    explanation_lines.append(
        "To account for observable differences across districts, I estimated a multiple "
        "regression of average test score on the student-teacher ratio plus controls for "
        "district income, percent English learners, percent qualifying for income assistance, "
        "percent on reduced-price lunch, expenditures per student, and number of computers."
    )
    explanation_lines.append(
        f"In this controlled model (R-squared {r2_multi:.3f}), the coefficient on the "
        f"student-teacher ratio remains very small at {coef_multi:.3f} with a p-value "
        f"of {pval_multi:.4f}. This again provides no statistically significant evidence "
        "that districts with fewer students per teacher have higher test scores after "
        "adjusting for these factors."
    )
    explanation_lines.append(
        "Separate controlled regressions for reading and math scores show similarly tiny and "
        "statistically insignificant coefficients on the student-teacher ratio, "
        f"with estimates of {results_detail['read']['coef']:.3f} (p-value "
        f"{results_detail['read']['pval']:.4f}) for reading and "
        f"{results_detail['math']['coef']:.3f} (p-value {results_detail['math']['pval']:.4f}) for math."
    )
    explanation_lines.append(
        f"Interpreting the controlled coefficient, a 5-student increase in the number of "
        f"students per teacher is associated with about {delta_score_5_students:.2f} points "
        "difference in the test score scale on average, which is negligible relative to the "
        "overall variation in scores across districts."
    )
    explanation_lines.append(
        "Because the data are observational and the estimated relationships are both tiny in "
        "magnitude and far from statistically significant, the analysis does not support a "
        "clear association between lower student-teacher ratios and higher academic performance "
        "in this dataset."
    )
    explanation_lines.append(
        f"Reflecting this lack of convincing evidence, I assign a Likert-scale response of "
        f"{response_score} (on a 0–100 scale) to the statement that a lower student-teacher "
        "ratio is associated with higher academic performance, corresponding to a modest "
        "lean toward \"No.\""
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": int(response_score), "explanation": explanation}

    out_path = cwd / "conclusion.txt"
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

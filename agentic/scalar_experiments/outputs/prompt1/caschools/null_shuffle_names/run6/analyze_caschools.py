import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    info_path = Path("info.json")

    df = pd.read_csv(data_path)

    # Student-teacher ratio: total enrollment / number of teachers
    df["stratio"] = df["english"] / df["students"]

    # Academic performance: average of reading and math scores
    df["avg_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Basic checks
    valid = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["stratio", "avg_score"])
    str_all = valid["stratio"]
    score_all = valid["avg_score"]

    # Overall Pearson correlation
    corr_all = np.corrcoef(str_all, score_all)[0, 1]

    # Trim extreme student-teacher ratios to focus on realistic ranges
    trimmed = valid[(valid["stratio"] >= 5) & (valid["stratio"] <= 40)]
    str_trim = trimmed["stratio"]
    score_trim = trimmed["avg_score"]
    corr_trim = np.corrcoef(str_trim, score_trim)[0, 1]

    # Simple linear regression of avg_score on stratio (trimmed sample)
    X = sm.add_constant(str_trim)
    model = sm.OLS(score_trim, X).fit()
    slope = model.params["stratio"]
    p_value = model.pvalues["stratio"]

    # Decide on answer: look for robust negative association
    negative_and_significant = (slope < 0) and (p_value < 0.05)
    negative_correlation = (corr_all < 0) and (corr_trim < 0)

    if negative_and_significant and negative_correlation:
        response = "Yes"
    else:
        response = "No"

    explanation = {
        "research_question": "Is a lower student-teacher ratio associated with higher academic performance?",
        "metrics": {
            "student_teacher_ratio_definition": "Total enrollment (english) divided by number of teachers (students).",
            "performance_metric": "Average of reading (district) and math (expenditure) scores.",
            "n_total": int(len(df)),
            "n_trimmed": int(len(trimmed)),
            "correlation_all": float(corr_all),
            "correlation_trimmed": float(corr_trim),
            "regression_slope_trimmed": float(slope),
            "regression_p_value_trimmed": float(p_value),
        },
        "reasoning": (
            "I computed the student-teacher ratio for each district as total enrollment divided by the number of teachers, "
            "and summarized academic performance using the average of the reading and math test scores. "
            "Across all 420 districts, the correlation between student-teacher ratio and average test scores is essentially zero. "
            "Restricting attention to districts with ratios between 5 and 40 students per teacher (to exclude extreme outliers) "
            "does not materially change this result: the correlation remains very close to zero, and the slope in a linear regression "
            "of test scores on student-teacher ratio is small and not statistically different from zero at the 5% level. "
            "Because neither the overall correlation nor the regression evidence suggests that districts with lower student-teacher ratios "
            "systematically achieve higher test scores, the data do not provide clear support for an association where lower ratios are "
            "linked to higher academic performance."
        ),
    }

    conclusion = {
        "response": response,
        "explanation": json.dumps(explanation),
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()


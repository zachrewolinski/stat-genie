import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Compute student-teacher ratio and an overall achievement measure.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing or problematic values.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["stratio", "avg_score", "income", "english", "calworks", "lunch"]
    )

    # Correlation between class size (ratio) and performance.
    r, pval = stats.pearsonr(df["stratio"], df["avg_score"])

    # Linear regression with basic controls for demographics and income.
    y = df["avg_score"]
    X = df[["stratio", "income", "english", "calworks", "lunch"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    coef_stratio = model.params["stratio"]
    pval_stratio = model.pvalues["stratio"]

    # Determine answer: is a lower student-teacher ratio associated with higher performance?
    # That corresponds to a negative association between ratio and scores.
    negative_correlation = r < 0
    negative_coef = coef_stratio < 0
    significant = (pval < 0.05) and (pval_stratio < 0.05)

    if negative_correlation and negative_coef and significant:
        response = "Yes"
        confidence = 85
    elif negative_correlation and negative_coef:
        response = "Yes"
        confidence = 70
    else:
        response = "No"
        confidence = 70

    explanation_lines = []
    explanation_lines.append(
        "Using 420 California K-6 and K-8 districts, "
        "I computed the student-teacher ratio as students divided by teachers "
        "and an overall achievement score as the average of reading and math scores."
    )
    explanation_lines.append(
        f"The Pearson correlation between the student-teacher ratio and average test score "
        f"is {r:.3f} with p-value {pval:.3g}, indicating "
        + ("a statistically significant negative association." if (r < 0 and pval < 0.05) else "that the simple correlation is not strongly different from zero.")
    )
    explanation_lines.append(
        "I also estimated an OLS regression of average test score on the student-teacher ratio, "
        "controlling for district income, percent English learners, and poverty-related measures "
        "(CalWorks and reduced-price lunch). "
        f"In this model, the coefficient on the student-teacher ratio is {coef_stratio:.3f} "
        f"with p-value {pval_stratio:.3g}, "
        + ("showing that, after controlling for these demographics, higher ratios are associated with lower scores." if coef_stratio < 0 else "so the conditional association does not clearly show that smaller ratios are linked to higher scores.")
    )
    explanation_lines.append(
        "Combining the direction and significance of both the correlation and regression results, "
        "I assessed whether districts with fewer students per teacher tend to have higher achievement, "
        "and used this to decide on a binary Yes/No answer and confidence score."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()


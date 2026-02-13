import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # According to info.json, "english" is total enrollment and "students" is number of teachers.
    # Define student-teacher ratio as students per teacher.
    df["stratio"] = df["english"] / df["students"]

    # Academic performance measures: average reading and math scores.
    df["avg_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing values in key variables (should not occur, but keep analysis robust).
    key_cols = ["stratio", "district", "expenditure", "avg_score"]
    df_clean = df.dropna(subset=key_cols).copy()

    # Simple Pearson correlations between student-teacher ratio and performance.
    corr_read = df_clean["stratio"].corr(df_clean["district"])
    corr_math = df_clean["stratio"].corr(df_clean["expenditure"])
    corr_avg = df_clean["stratio"].corr(df_clean["avg_score"])

    # Regression of average test score on student-teacher ratio and key covariates.
    # Based on info.json descriptions:
    # - income: district average income (in thousands)
    # - school: percent qualifying for CalWorks (income assistance)
    # - computer: percent qualifying for reduced-price lunch
    # - rownames: percent of English learners
    # - grades: expenditure per student
    covariates = ["income", "school", "computer", "rownames", "grades"]
    available_covariates = [c for c in covariates if c in df_clean.columns]

    X = df_clean[["stratio"] + available_covariates]
    X = sm.add_constant(X)
    y = df_clean["avg_score"]

    model = sm.OLS(y, X).fit()

    coef_stratio = model.params["stratio"]
    pval_stratio = model.pvalues["stratio"]

    # Determine binary response based on sign and significance of association.
    # Negative coefficient / correlations indicate that lower ratio is associated with higher performance.
    negative_association = (coef_stratio < 0) and (corr_avg < 0)

    if negative_association:
        response = "Yes"
    else:
        response = "No"

    abs_corr = float(abs(corr_avg))

    # Strength and confidence heuristics depend on whether we see a clear negative association.
    if negative_association:
        # Evidence in favor of the hypothesis.
        if pval_stratio < 0.001 and abs_corr >= 0.3:
            strength = 90.0
            confidence = 85.0
        elif pval_stratio < 0.01 and abs_corr >= 0.2:
            strength = 80.0
            confidence = 80.0
        elif pval_stratio < 0.05 and abs_corr >= 0.1:
            strength = 70.0
            confidence = 75.0
        elif pval_stratio < 0.1 and abs_corr >= 0.1:
            strength = 60.0
            confidence = 70.0
        else:
            strength = 50.0
            confidence = 60.0
    else:
        # No clear negative association (either near zero or in the opposite direction).
        if abs_corr < 0.05 and pval_stratio > 0.3:
            strength = 85.0
            confidence = 75.0
        elif abs_corr < 0.1 and pval_stratio > 0.1:
            strength = 70.0
            confidence = 70.0
        else:
            strength = 55.0
            confidence = 60.0

    # Ensure values are within 0-100.
    strength = max(0.0, min(100.0, strength))
    confidence = max(0.0, min(100.0, confidence))

    # Build explanation text summarizing key numerical evidence.
    explanation_lines = []
    explanation_lines.append(
        "I defined the student-teacher ratio as total enrollment divided by the number of teachers "
        "(`english` / `students` in the dataset), and academic performance as the average of the "
        "district-level reading and math scores (`district` and `expenditure`)."
    )
    explanation_lines.append(
        f"The Pearson correlation between the student-teacher ratio and average test score "
        f"is {corr_avg:.3f}, with correlations of {corr_read:.3f} and {corr_math:.3f} for "
        f"reading and math scores separately."
    )
    explanation_lines.append(
        "I then fit a linear regression of the average test score on the student-teacher ratio, "
        "controlling for district income, poverty indicators (CalWorks and reduced-price lunch), "
        "percent English learners, and per-pupil expenditure."
    )
    if negative_association:
        explanation_lines.append(
            f"In this regression, the coefficient on the student-teacher ratio is {coef_stratio:.3f} "
            f"with a p-value of {pval_stratio:.4f}. The negative coefficient and negative correlations "
            f"indicate that districts with fewer students per teacher tend to have higher test scores, "
            f"after accounting for observed covariates."
        )
        explanation_lines.append(
            "Because the data are observational and cross-sectional, this establishes an association "
            "rather than a definitive causal effect, but the direction and magnitude of the estimates "
            "are consistent with the hypothesis that lower student-teacher ratios are linked to better "
            "academic performance."
        )
    else:
        explanation_lines.append(
            f"In this regression, the coefficient on the student-teacher ratio is {coef_stratio:.3f} "
            f"with a p-value of {pval_stratio:.4f}. The coefficient is very close to zero and not "
            f"statistically significant, and the correlations are also near zero, indicating little "
            f"systematic relationship between the student-teacher ratio and test scores in this sample "
            f"once covariates are included."
        )
        explanation_lines.append(
            "Given the observational, cross-sectional nature of the data, we cannot rule out small effects "
            "or unmeasured confounding, but the available evidence does not support a strong association "
            "between lower student-teacher ratios and higher academic performance in these districts."
        )

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": response,
        "strength": round(float(strength), 1),
        "confidence": round(float(confidence), 1),
        "explanation": explanation,
    }

    # Write required JSON object to conclusion.txt with no extra text.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df = df.copy()
    df["student_teacher_ratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key values, if present
    df = df.dropna(subset=["student_teacher_ratio", "testscr"])

    # Pearson correlation between ratio and test score
    r, p_value = stats.pearsonr(df["student_teacher_ratio"], df["testscr"])

    # Simple OLS regression: testscr ~ ratio
    X = sm.add_constant(df["student_teacher_ratio"])
    model = sm.OLS(df["testscr"], X).fit()
    coef_ratio = model.params["student_teacher_ratio"]
    coef_p_value = model.pvalues["student_teacher_ratio"]
    r_squared = model.rsquared

    # Direction: lower ratio (fewer students per teacher) should be associated
    # with higher test scores, which corresponds to a negative slope.
    has_evidence = coef_p_value < 0.05 and coef_ratio < 0

    # Map strength of association to a 0–100 Likert-style score.
    # Use magnitude of correlation and R^2, but cap within [0, 1].
    effect_strength = float(abs(r))
    r2_strength = float(r_squared)
    combined_strength = min(1.0, 0.6 * effect_strength + 0.4 * r2_strength)

    if has_evidence:
        # Base confidence 60 for significant result, plus scaled strength.
        response_score = 60 + int(round(40 * combined_strength))
    else:
        # No significant evidence: score closer to 0, but allow moderate values
        # if the point estimates are reasonably aligned with the hypothesis.
        if coef_ratio < 0:
            # Right direction but not significant
            response_score = max(0, 30 - int(round(20 * (1 - combined_strength))))
        else:
            # Wrong direction or null
            response_score = 10

    response_score = int(max(0, min(100, response_score)))

    # Build a concise, human-readable explanation.
    direction_text = (
        "lower student-teacher ratios are associated with higher test scores"
        if coef_ratio < 0
        else "higher student-teacher ratios are (counterintuitively) associated with higher test scores"
    )

    significance_text = (
        f"The OLS slope for the student-teacher ratio is {coef_ratio:.3f} "
        f"(p = {coef_p_value:.4g}), with R² = {r_squared:.3f}. "
        f"The Pearson correlation between the ratio and average test score is r = {r:.3f} (p = {p_value:.4g}). "
    )

    if has_evidence:
        answer_text = (
            "These results provide statistically significant evidence that "
            f"{direction_text} in this sample."
        )
    else:
        answer_text = (
            "These results do not provide statistically significant evidence that "
            "lower student-teacher ratios are associated with higher academic performance."
        )

    explanation = significance_text + answer_text

    conclusion = {"response": response_score, "explanation": explanation}

    # Write JSON output exactly as required
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


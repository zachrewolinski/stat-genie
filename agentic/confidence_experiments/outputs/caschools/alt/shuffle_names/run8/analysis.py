import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Decode true variable semantics based on the metadata and known structure.
    # english -> total enrollment, students -> number of teachers (FTE)
    # district -> average reading score, expenditure -> average math score
    df["enrollment"] = df["english"]
    df["teachers_fte"] = df["students"]
    df["stratio"] = df["enrollment"] / df["teachers_fte"]

    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["avg_score"] = df[["read_score", "math_score"]].mean(axis=1)

    # Socio-economic and demographic controls
    df["income_thousands"] = df["income"]
    df["pct_english_learners"] = df["rownames"]
    df["pct_lunch"] = df["computer"]

    # Drop any rows with missing key values (should be none, but be safe).
    model_df = df[
        [
            "stratio",
            "avg_score",
            "read_score",
            "math_score",
            "income_thousands",
            "pct_english_learners",
            "pct_lunch",
        ]
    ].dropna()

    # Simple bivariate association
    corr_avg = model_df["stratio"].corr(model_df["avg_score"])
    corr_read = model_df["stratio"].corr(model_df["read_score"])
    corr_math = model_df["stratio"].corr(model_df["math_score"])

    # Regression of average score on student–teacher ratio
    simple_model = smf.ols("avg_score ~ stratio", data=model_df).fit()

    # Multiple regression with key controls
    full_model = smf.ols(
        "avg_score ~ stratio + income_thousands + pct_english_learners + pct_lunch",
        data=model_df,
    ).fit()

    # Extract key statistics for interpretation
    simple_coef = simple_model.params["stratio"]
    simple_p = simple_model.pvalues["stratio"]

    full_coef = full_model.params["stratio"]
    full_p = full_model.pvalues["stratio"]

    # Decide on strength of evidence:
    # - Require p < 0.05 in both models for a clear "Yes"
    # - The more negative and statistically significant the coefficient, the higher the score.
    if (simple_p < 0.05) and (full_p < 0.05) and (simple_coef < 0) and (full_coef < 0):
        # Scale likelihood based on magnitude of the effect in the full model.
        # Typical STR is around 20; use effect size per student as heuristic.
        effect_per_student = abs(full_coef)
        # Map roughly: 1 point per student -> very strong (around 95),
        # 0.5 -> strong (around 85), 0.2 -> moderate (around 70).
        if effect_per_student >= 1.0:
            response_score = 95
        elif effect_per_student >= 0.5:
            response_score = 85
        elif effect_per_student >= 0.2:
            response_score = 70
        else:
            response_score = 60
    elif (simple_p < 0.1) and (full_p < 0.1) and (simple_coef < 0) and (full_coef < 0):
        response_score = 55
    else:
        # Little or no evidence that lower ratios are associated with higher performance.
        response_score = 30

    # Build a concise explanation summarizing the evidence.
    explanation = {
        "research_question": "Is a lower student-teacher ratio associated with higher academic performance?",
        "data_summary": {
            "n_districts": int(model_df.shape[0]),
            "corr_avg_score_vs_ratio": float(corr_avg),
            "corr_read_vs_ratio": float(corr_read),
            "corr_math_vs_ratio": float(corr_math),
        },
        "simple_model": {
            "formula": "avg_score ~ stratio",
            "coef_stratio": float(simple_coef),
            "p_value_stratio": float(simple_p),
            "r_squared": float(simple_model.rsquared),
        },
        "full_model": {
            "formula": "avg_score ~ stratio + income + pct_english_learners + pct_lunch",
            "coef_stratio": float(full_coef),
            "p_value_stratio": float(full_p),
            "r_squared": float(full_model.rsquared),
        },
        "interpretation": (
            "Negative coefficients for the student-teacher ratio in both models, combined with "
            "statistical significance, indicate that districts with fewer students per teacher "
            "tend to have higher average test scores. The association remains after adjusting for "
            "income and student demographics, suggesting a meaningful relationship rather than a "
            "spurious correlation."
            if response_score >= 55
            else "Across correlation and regression analyses, we do not find consistent, statistically "
            "robust evidence that districts with fewer students per teacher have higher test scores "
            "once key covariates are taken into account."
        ),
    }

    conclusion = {"response": int(response_score), "explanation": json.dumps(explanation)}

    # Write JSON object exactly as required.
    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


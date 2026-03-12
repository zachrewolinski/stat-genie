import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename to semantic meanings for clarity
    df = df.rename(
        columns={
            "english": "enrollment",  # Total enrollment
            "students": "n_teachers",  # Number of teachers
            "school": "pct_calworks",
            "computer": "pct_lunch",
            "county": "computers",
            "grades": "exp_per_student",
            "income": "avg_income",
            "rownames": "pct_english_learners",
            "district": "avg_read",
            "expenditure": "avg_math",
        }
    )

    # Construct key variables
    df["student_teacher_ratio"] = df["enrollment"] / df["n_teachers"]
    df["avg_score"] = (df["avg_read"] + df["avg_math"]) / 2.0

    # Basic cleaning: drop impossible or missing ratios
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["student_teacher_ratio", "avg_score", "avg_income", "exp_per_student"]
    )
    df = df[df["n_teachers"] > 0]
    df = df[df["enrollment"] > 0]

    return df


def correlation_analysis(df: pd.DataFrame) -> dict:
    r, p = stats.pearsonr(df["student_teacher_ratio"], df["avg_score"])
    return {"pearson_r": float(r), "p_value": float(p)}


def regression_analysis(df: pd.DataFrame) -> dict:
    # Simple bivariate regression
    X_simple = sm.add_constant(df["student_teacher_ratio"])
    model_simple = sm.OLS(df["avg_score"], X_simple).fit()

    # Multiple regression with key controls
    controls = [
        "avg_income",
        "exp_per_student",
        "pct_calworks",
        "pct_lunch",
        "pct_english_learners",
    ]
    X_controls = df[["student_teacher_ratio"] + controls].copy()
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df["avg_score"], X_controls).fit()

    return {
        "simple_coef": float(model_simple.params["student_teacher_ratio"]),
        "simple_p": float(model_simple.pvalues["student_teacher_ratio"]),
        "simple_r2": float(model_simple.rsquared),
        "controls_coef": float(model_controls.params["student_teacher_ratio"]),
        "controls_p": float(model_controls.pvalues["student_teacher_ratio"]),
        "controls_r2": float(model_controls.rsquared),
    }


def infer_likert_score(corr: dict, reg: dict) -> tuple[int, str]:
    """
    Map statistical evidence to a 0–100 Likert score and explanation.
    """
    r = corr["pearson_r"]
    p_corr = corr["p_value"]
    coef_simple = reg["simple_coef"]
    p_simple = reg["simple_p"]
    coef_ctrl = reg["controls_coef"]
    p_ctrl = reg["controls_p"]

    # Direction: negative slope means lower ratios -> higher scores
    direction_consistent = (coef_simple < 0) and (coef_ctrl < 0) and (r < 0)
    significant = (p_corr < 0.05) and (p_simple < 0.05) and (p_ctrl < 0.05)

    # Magnitude heuristics based on standardized effect sizes
    abs_r = abs(r)
    if significant and direction_consistent:
        if abs_r >= 0.5:
            score = 90
        elif abs_r >= 0.3:
            score = 80
        elif abs_r >= 0.2:
            score = 70
        else:
            score = 60
    elif direction_consistent and ((p_corr < 0.1) or (p_simple < 0.1) or (p_ctrl < 0.1)):
        score = 55
    else:
        # No reliable evidence of association
        if abs_r <= 0.1 and not significant:
            score = 20
        else:
            score = 40

    # Clamp to required range and cast to int
    score_int = int(max(0, min(100, round(score))))

    # Build textual summary
    explanation = (
        "We examined whether districts with lower student-teacher ratios tend to "
        "have higher average test scores. The student-teacher ratio was defined as "
        "total enrollment divided by the number of teachers; academic performance "
        "was measured as the average of district-level reading and math scores.\n\n"
        f"The Pearson correlation between student-teacher ratio and average score "
        f"was {r:.3f} (p-value {p_corr:.3g}), indicating a "
    )

    if r < 0:
        explanation += "negative "
    elif r > 0:
        explanation += "positive "
    else:
        explanation += "near-zero "

    explanation += "association.\n\n"

    explanation += (
        "A simple linear regression of average score on the student-teacher ratio "
        f"yielded a coefficient of {coef_simple:.3f} (p-value {p_simple:.3g}, "
        f"R-squared {reg['simple_r2']:.3f}). "
        "We then estimated a multiple regression controlling for average district "
        "income, expenditure per student, and the shares of students on income "
        "assistance, qualifying for reduced-price lunch, and classified as "
        "English learners. In this model, the coefficient on the student-teacher "
        f"ratio was {coef_ctrl:.3f} (p-value {p_ctrl:.3g}, "
        f"R-squared {reg['controls_r2']:.3f}).\n\n"
    )

    if significant and direction_consistent:
        explanation += (
            "Across these analyses, the student-teacher ratio is consistently "
            "negatively and statistically significantly related to test scores, "
            "even after adjusting for key socioeconomic and demographic factors. "
            "This provides reasonably strong evidence that districts with lower "
            "student-teacher ratios tend to have higher academic performance, "
            "though the association is observational and should not be interpreted "
            "as strictly causal."
        )
    elif direction_consistent:
        explanation += (
            "The estimated relationships are generally in the expected negative "
            "direction (lower ratios associated with higher scores), but the "
            "statistical evidence is weaker, with some coefficients only "
            "marginally significant. This suggests a possible association, but "
            "the evidence is not strong."
        )
    else:
        explanation += (
            "The estimated relationships are not consistently in the expected "
            "direction or are not statistically significant. Overall, the data "
            "do not provide strong evidence that lower student-teacher ratios "
            "are associated with higher academic performance after accounting "
            "for other factors."
        )

    explanation += (
        f"\n\nOn a 0–100 scale where 0 is a strong 'No' "
        f"and 100 is a strong 'Yes', a score of {score_int} "
        "summarizes the strength of evidence that lower student-teacher ratios "
        "are associated with higher academic performance in this dataset."
    )

    return score_int, explanation


def main() -> None:
    csv_path = Path("caschools.csv")
    df = load_data(csv_path)

    corr = correlation_analysis(df)
    reg = regression_analysis(df)
    score, explanation = infer_likert_score(corr, reg)

    conclusion = {"response": score, "explanation": explanation}

    out_path = Path("conclusion.txt")
    with out_path.open("w") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

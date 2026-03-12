import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def construct_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Student–teacher ratio: enrollment / number of teachers
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    return df


def simple_correlation(df: pd.DataFrame):
    r, p = stats.pearsonr(df["student_teacher_ratio"], df["testscr"])
    return r, p


def simple_regression(df: pd.DataFrame):
    X = sm.add_constant(df["student_teacher_ratio"])
    model = sm.OLS(df["testscr"], X).fit()
    coef = model.params["student_teacher_ratio"]
    pval = model.pvalues["student_teacher_ratio"]
    r2 = model.rsquared
    return coef, pval, r2


def multiple_regression(df: pd.DataFrame):
    covariates = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    cols = ["student_teacher_ratio"] + covariates
    clean = df[cols + ["testscr"]].dropna()
    X = sm.add_constant(clean[cols])
    model = sm.OLS(clean["testscr"], X).fit()
    coef = model.params["student_teacher_ratio"]
    pval = model.pvalues["student_teacher_ratio"]
    r2 = model.rsquared
    return coef, pval, r2


def compute_likert_score(r: float, p: float, coef_multi: float, p_multi: float) -> int:
    """
    Map evidence of association to a 0–100 Likert scale.
    Higher scores correspond to stronger evidence that
    lower student–teacher ratios are associated with higher performance.
    """
    # Default: ambiguous/weak evidence
    score = 50

    # Direction: we expect a negative relationship between ratio and scores.
    if p >= 0.05:
        # No statistically significant simple association
        if r < 0:
            score = 40
        else:
            score = 30
    else:
        # Significant simple association; scale with effect size
        magnitude = abs(r)
        if magnitude < 0.1:
            score = 60
        elif magnitude < 0.3:
            score = 70
        elif magnitude < 0.5:
            score = 80
        else:
            score = 90

    # Strengthen confidence if multivariable model agrees (negative and significant)
    if p_multi < 0.05 and coef_multi < 0:
        score = min(100, score + 5)

    # Ensure integer in [0, 100]
    score = int(round(max(0, min(100, score))))
    return score


def main() -> None:
    csv_path = Path("caschools.csv")
    df_raw = load_data(csv_path)
    df = construct_variables(df_raw)

    # Core analyses
    r, p = simple_correlation(df)
    coef_simple, p_simple, r2_simple = simple_regression(df)
    coef_multi, p_multi, r2_multi = multiple_regression(df)

    # Compute a Likert-style confidence score
    likert = compute_likert_score(r, p, coef_multi, p_multi)

    # Translate Likert score into a plain-language Yes/No
    qualitative = "Yes" if likert > 50 else "No"

    # Express change for a 5-student reduction in ratio using the multivariable model
    effect_per_student = coef_multi
    effect_for_five_students = effect_per_student * -5.0  # reduction in ratio (negative change)

    explanation = (
        "Using data from 420 California K-6 and K-8 school districts, "
        "I constructed the student-teacher ratio as total enrollment divided by the number of teachers "
        "(feature6/feature7) and defined academic performance as the average of district reading and math scores "
        "(mean of feature14 and feature15). "
        "The simple Pearson correlation between the student-teacher ratio and test scores is "
        f"r={r:.3f} with p-value={p:.2e}, indicating that districts with more students per teacher tend to have "
        "lower average test scores when this correlation is negative and statistically significant. "
        "A simple linear regression of test scores on the student-teacher ratio yields a coefficient of "
        f"{coef_simple:.3f} (p-value={p_simple:.2e}, R-squared={r2_simple:.3f}), "
        "so each additional student per teacher is associated with this many points difference in average test scores. "
        "To account for key demographic and resource factors, I estimated a multiple regression including the "
        "student-teacher ratio along with CalWorks percentage (feature8), reduced-price lunch percentage (feature9), "
        "expenditure per student (feature11), district average income (feature12), and the percentage of English "
        f"learners (feature13). In this multivariable model, the coefficient on the student-teacher ratio is "
        f"{coef_multi:.3f} (p-value={p_multi:.2e}, R-squared={r2_multi:.3f}). "
        f"A reduction of 5 students per teacher is associated with an estimated change of "
        f"{effect_for_five_students:.2f} points in average test scores, holding these covariates constant. "
        f"Taken together, these results support the conclusion '{qualitative}' to the question "
        "'Is a lower student-teacher ratio associated with higher academic performance?', "
        "and I summarize the overall strength of evidence for this association with a Likert-style score of "
        f"{likert} on a 0-to-100 scale, where higher values indicate stronger evidence for a positive association."
    )

    result = {"response": likert, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


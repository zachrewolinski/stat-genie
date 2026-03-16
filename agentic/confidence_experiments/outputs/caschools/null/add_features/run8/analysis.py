import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_PATH = Path("caschools.csv")
CONCLUSION_PATH = Path("conclusion.txt")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Student–teacher ratio: students per teacher (class size proxy)
    df["stratio"] = df["students"] / df["teachers"]
    # Academic performance: average of reading and math scores
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    return df


def run_regressions(df: pd.DataFrame):
    """Run bivariate and multivariate regressions of testscr on stratio."""
    # Drop any rows with missing values in key variables (there should be none, but be safe).
    base_cols = ["testscr", "stratio", "calworks", "lunch", "english", "income", "expenditure"]
    reg_df = df[base_cols].dropna().copy()

    y = reg_df["testscr"]
    x_biv = sm.add_constant(reg_df["stratio"])
    model_biv = sm.OLS(y, x_biv).fit()

    # Multivariate model controlling for key socioeconomic and resource covariates.
    x_multi = reg_df[["stratio", "calworks", "lunch", "english", "income", "expenditure"]]
    x_multi = sm.add_constant(x_multi)
    model_multi = sm.OLS(y, x_multi).fit()

    return model_biv, model_multi


def compute_likert_from_model(model) -> int:
    """
    Map the estimated effect of stratio on testscr to a 0–100 Likert score.

    Interpretation:
    - Negative coefficient on stratio means lower student–teacher ratio
      (smaller classes) is associated with higher test scores.
    - We combine sign, magnitude, and statistical significance into a scalar.
    """
    beta = model.params["stratio"]
    pval = model.pvalues["stratio"]

    # If effect clearly in the wrong direction and significant, strong "No".
    if beta > 0 and pval < 0.05:
        # Higher ratios (larger classes) associated with higher performance.
        return 5

    # If effect not statistically distinguishable from zero, lean toward "No".
    if pval >= 0.1:
        return 30
    if 0.05 <= pval < 0.1:
        return 45

    # At this point pval < 0.05, so there is statistically significant evidence of a relationship.
    # Direction matters: if beta < 0, this supports the research hypothesis.
    if beta >= 0:
        # Significant but in the opposite direction -> strong "No".
        return 10

    # Significant and in the hypothesized (negative) direction: map strength.
    # Base score for a statistically significant negative association.
    score = 60

    # Significance bonus.
    if pval < 0.001:
        score += 20
    elif pval < 0.01:
        score += 15
    else:  # 0.01 <= pval < 0.05
        score += 10

    # Magnitude bonus: typical SD of testscr ~ 20.
    # A 1-student increase per teacher changing scores by ~2 points is sizable.
    mag_bonus = min(abs(beta) * 5.0, 20.0)
    score += mag_bonus

    # Ensure integer between 0 and 100.
    score = int(round(np.clip(score, 0, 100)))
    return score


def build_explanation(model_biv, model_multi, likert_score: int) -> str:
    beta_biv = model_biv.params["stratio"]
    p_biv = model_biv.pvalues["stratio"]
    r2_biv = model_biv.rsquared

    beta_multi = model_multi.params["stratio"]
    p_multi = model_multi.pvalues["stratio"]
    r2_multi = model_multi.rsquared

    direction = "decrease" if beta_multi < 0 else "increase"

    lines = []
    lines.append(
        "Research question: Is a lower student–teacher ratio associated with higher academic performance "
        "in California K–8 school districts (5th grade Stanford 9 scores)?"
    )
    lines.append(
        "I used the 1998–1999 district-level caschools data (420 districts), "
        "defining the student–teacher ratio as students per teacher and academic performance "
        "as the average of reading and math test scores."
    )
    lines.append(
        f"In a simple regression of average test score on the student–teacher ratio, "
        f"an additional student per teacher is associated with a change of approximately "
        f"{beta_biv:.2f} points in the average test score (p = {p_biv:.4f}, R² = {r2_biv:.3f})."
    )
    lines.append(
        "To account for demographic and resource differences across districts, I estimated a multiple "
        "regression of average test scores on the student–teacher ratio, controlling for the percentages "
        "of students on CalWorks and reduced-price lunch, the percentage of English learners, average income, "
        "and per-pupil expenditures."
    )
    lines.append(
        f"In this multivariate model, a one-student increase in the student–teacher ratio is associated with "
        f"an average {direction} of {abs(beta_multi):.2f} points in test scores "
        f"(p = {p_multi:.4f}, R² = {r2_multi:.3f})."
    )

    if beta_multi < 0 and p_multi < 0.05:
        lines.append(
            "The negative and statistically significant coefficient on the student–teacher ratio indicates that, "
            "after adjusting for key socioeconomic and resource covariates, districts with smaller classes tend "
            "to have higher test scores. The magnitude suggests that moving from a relatively large to a "
            "moderate class size can shift average test performance by several points, which is meaningful "
            "given that the overall score standard deviation is around 20 points."
        )
        lines.append(
            f"Given the direction, magnitude, and statistical significance of the estimated effect, I interpret "
            f"this as clear evidence of an association between lower student–teacher ratios and higher academic "
            f"performance in this observational dataset. On a 0–100 Likert scale, I would summarize my answer as "
            f"a 'Yes' with strength {likert_score}."
        )
    elif p_multi >= 0.05:
        lines.append(
            "In the multivariate model the coefficient on the student–teacher ratio is not statistically "
            "distinguishable from zero at conventional significance levels, meaning the data do not provide "
            "strong evidence that class size is related to test performance once other factors are taken into account."
        )
        lines.append(
            f"As a result, I interpret the evidence as insufficient to claim a clear association between lower "
            f"student–teacher ratios and higher academic performance. On a 0–100 Likert scale, I would summarize "
            f"my answer as a cautious 'No' with strength {likert_score}."
        )
    else:
        lines.append(
            "Although the estimated coefficient on the student–teacher ratio is statistically significant, it is "
            "in the direction opposite to the hypothesized relationship, suggesting that smaller classes are not "
            "associated with higher test scores in this specification."
        )
        lines.append(
            f"Consequently, I interpret the data as evidence against the hypothesized relationship. On a 0–100 "
            f"Likert scale, I would summarize my answer as a 'No' with strength {likert_score}."
        )

    return " ".join(lines)


def main():
    df = load_data(DATA_PATH)
    model_biv, model_multi = run_regressions(df)
    likert_score = compute_likert_from_model(model_multi)
    explanation = build_explanation(model_biv, model_multi, likert_score)

    result = {"response": int(likert_score), "explanation": explanation}

    # Write JSON with no extra whitespace or lines beyond the single object.
    CONCLUSION_PATH.write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


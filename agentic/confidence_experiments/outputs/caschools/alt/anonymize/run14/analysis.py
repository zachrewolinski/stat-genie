import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio (students per teacher).
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores.
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)

    # Drop any rows with missing values in variables of interest (should be none, but safe).
    analysis_df = df[
        [
            "stratio",
            "avg_score",
            "feature8",
            "feature9",
            "feature11",
            "feature12",
            "feature13",
        ]
    ].dropna()

    # Correlation between student-teacher ratio and average test score.
    r, p_value = stats.pearsonr(analysis_df["stratio"], analysis_df["avg_score"])

    # Simple linear regression: avg_score ~ stratio.
    model_simple = smf.ols("avg_score ~ stratio", data=analysis_df).fit()
    beta_simple = float(model_simple.params["stratio"])
    beta_simple_p = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Adjusted regression controlling for key demographics and resources.
    model_adj = smf.ols(
        "avg_score ~ stratio + feature8 + feature9 + feature11 + feature12 + feature13",
        data=analysis_df,
    ).fit()
    beta_adj = float(model_adj.params["stratio"])
    beta_adj_p = float(model_adj.pvalues["stratio"])
    r2_adj = float(model_adj.rsquared)

    # Effect over interquartile range of student-teacher ratio using adjusted model.
    stratio_q1 = float(analysis_df["stratio"].quantile(0.25))
    stratio_q3 = float(analysis_df["stratio"].quantile(0.75))

    controls_mean = {
        "feature8": float(analysis_df["feature8"].mean()),
        "feature9": float(analysis_df["feature9"].mean()),
        "feature11": float(analysis_df["feature11"].mean()),
        "feature12": float(analysis_df["feature12"].mean()),
        "feature13": float(analysis_df["feature13"].mean()),
    }

    pred_low = float(
        model_adj.predict(
            {
                "stratio": [stratio_q1],
                **{k: [v] for k, v in controls_mean.items()},
            }
        )[0]
    )
    pred_high = float(
        model_adj.predict(
            {
                "stratio": [stratio_q3],
                **{k: [v] for k, v in controls_mean.items()},
            }
        )[0]
    )
    iqr_effect = pred_low - pred_high  # positive if lower ratio -> higher scores

    # Map evidence to a 0-100 Likert-style score.
    response_score = score_evidence(r, p_value)

    explanation = build_explanation(
        r=r,
        p_value=p_value,
        beta_simple=beta_simple,
        beta_simple_p=beta_simple_p,
        r2_simple=r2_simple,
        beta_adj=beta_adj,
        beta_adj_p=beta_adj_p,
        r2_adj=r2_adj,
        stratio_q1=stratio_q1,
        stratio_q3=stratio_q3,
        iqr_effect=iqr_effect,
        response_score=response_score,
    )

    result = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


def score_evidence(r: float, p_value: float) -> int:
    """
    Convert correlation and its p-value into a 0-100 Likert score.

    Values above 50 correspond to "Yes, there is an association"
    (here: lower student-teacher ratio associated with higher performance),
    and values below 50 correspond to "No".
    """
    # Base on significance.
    if p_value >= 0.1:
        base = 30
    elif p_value >= 0.05:
        base = 40
    elif p_value >= 0.01:
        base = 60
    elif p_value >= 0.001:
        base = 75
    else:
        base = 85

    # Adjust for effect size magnitude.
    abs_r = abs(r)
    if abs_r < 0.1:
        effect_adj = -10
    elif abs_r < 0.3:
        effect_adj = 0
    elif abs_r < 0.5:
        effect_adj = 5
    else:
        effect_adj = 10

    score = base + effect_adj

    # Ensure direction matches the research question:
    # We expect a negative correlation (higher ratio -> lower performance).
    # If correlation is positive, evidence goes against the hypothesis.
    if r > 0:
        score = 100 - score

    score = int(np.clip(round(score), 0, 100))
    return score


def build_explanation(
    r: float,
    p_value: float,
    beta_simple: float,
    beta_simple_p: float,
    r2_simple: float,
    beta_adj: float,
    beta_adj_p: float,
    r2_adj: float,
    stratio_q1: float,
    stratio_q3: float,
    iqr_effect: float,
    response_score: int,
) -> str:
    direction_phrase = (
        "districts with fewer students per teacher tend to have higher test scores"
        if r < 0
        else "districts with more students per teacher tend to have higher test scores"
    )

    conclusion_phrase = (
        "Overall, this provides strong evidence that lower student-teacher ratios are associated with higher academic performance."
        if response_score >= 70 and r < 0 and p_value < 0.01
        else "Overall, this provides limited or mixed evidence for an association between student-teacher ratios and academic performance."
    )

    explanation = (
        "Research question: Is a lower student-teacher ratio associated with higher academic performance "
        "in California K-6 and K-8 school districts?\n\n"
        "Data and variables: The dataset contains 420 school districts with information on total enrollment "
        "(feature6), number of teachers (feature7), socioeconomic and demographic indicators "
        "(percent CalWorks, reduced-price lunch, expenditures per student, district income, percent English learners; "
        "features 8, 9, 11, 12, 13) and test scores (average reading and math scores; features 14 and 15). "
        "I constructed the student-teacher ratio as students per teacher (feature6 / feature7) and "
        "measured academic performance as the average of reading and math scores.\n\n"
        "Correlation analysis: The Pearson correlation between student-teacher ratio and average test score is "
        f"{r:.3f} (p = {p_value:.3g}). This means {direction_phrase}. "
        "The magnitude of the correlation quantifies how strongly districts with lower ratios tend to perform better academically.\n\n"
        "Regression analysis (unadjusted): A simple linear regression of average test score on student-teacher ratio "
        f"yields an estimated coefficient of {beta_simple:.2f} points per additional student per teacher "
        f"(p = {beta_simple_p:.3g}, R² = {r2_simple:.3f}). A negative coefficient indicates that larger classes "
        "are associated with lower test scores.\n\n"
        "Regression analysis (adjusted for confounders): To account for socioeconomic and demographic differences, "
        "I estimated a multiple regression including percent CalWorks, percent reduced-price lunch, expenditures per student, "
        "district income, and percent English learners as controls. In this model, the coefficient on student-teacher ratio is "
        f"{beta_adj:.2f} (p = {beta_adj_p:.3g}, R² = {r2_adj:.3f}). The sign and significance of this coefficient show whether "
        "the association between class size and performance persists after adjusting for these background factors.\n\n"
        "Practical magnitude: Comparing districts at the 25th percentile of the student-teacher ratio "
        f"({stratio_q1:.1f} students per teacher) to those at the 75th percentile "
        f"({stratio_q3:.1f} students per teacher), while holding other variables at their means, the adjusted model "
        f"predicts a difference of about {iqr_effect:.2f} points in average test scores (lower ratios implying higher scores "
        "when this difference is positive). This represents the typical performance advantage associated with moving from "
        "relatively larger to relatively smaller classes.\n\n"
        f"Mapped to a 0–100 Likert scale, the strength of evidence that lower student-teacher ratios are associated with higher academic performance is summarized by a score of {response_score}. "
        f"{conclusion_phrase}"
    )

    return explanation


if __name__ == "__main__":
    main()


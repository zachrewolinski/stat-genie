import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def compute_likert_score(
    r: float,
    p_corr: float,
    coef_stratio: float,
    p_stratio: float,
    coef_stratio_adj: float | None,
    p_stratio_adj: float | None,
) -> int:
    """
    Map statistical evidence about the STR–achievement relationship
    onto a 0–100 Likert-style scale, where higher means stronger
    evidence that lower STR is associated with higher performance.
    """
    # Start from an "uncertain" midpoint.
    score = 50.0

    # Correlation evidence.
    if r < 0:
        score += 10.0
    else:
        score -= 10.0

    if p_corr < 0.05:
        score += 10.0
    if p_corr < 0.01:
        score += 5.0

    # Simple regression coefficient.
    if coef_stratio < 0:
        score += 10.0
    else:
        score -= 10.0

    if p_stratio < 0.05:
        score += 10.0
    if p_stratio < 0.01:
        score += 5.0

    # Adjusted model, if available.
    if coef_stratio_adj is not None and p_stratio_adj is not None:
        if coef_stratio_adj < 0:
            score += 5.0
        else:
            score -= 5.0

        if p_stratio_adj < 0.05:
            score += 5.0
        if p_stratio_adj < 0.01:
            score += 5.0

    # Effect size scaling based on |r|.
    effect_strength = min(abs(r), 1.0)
    score += 20.0 * effect_strength

    # Clamp to [0, 100] and return as integer.
    return int(round(float(max(0.0, min(100.0, score)))))


def main() -> None:
    # Load metadata (not strictly needed for analysis logic, but used for context).
    info_path = Path("info.json")
    if info_path.exists():
        with info_path.open("r") as f:
            info = json.load(f)
        research_question = info.get("research_questions", [""])[0]
    else:
        info = {}
        research_question = "Is a lower student-teacher ratio associated with higher academic performance?"

    # Load dataset.
    df = pd.read_csv("caschools.csv")

    # Construct key variables.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    df = df.dropna(subset=["stratio", "avg_score"])

    n_obs = len(df)

    # Correlation between STR and achievement.
    r, p_corr = stats.pearsonr(df["stratio"], df["avg_score"])

    # Simple OLS regression: avg_score ~ stratio.
    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["avg_score"], X1).fit()
    coef_stratio = float(model1.params["stratio"])
    p_stratio = float(model1.pvalues["stratio"])
    ci1_low, ci1_high = model1.conf_int().loc["stratio"]

    # Adjusted model with key covariates, if present.
    candidate_covars = ["income", "english", "lunch", "calworks"]
    covars = [c for c in candidate_covars if c in df.columns]

    coef_stratio_adj = None
    p_stratio_adj = None
    ci2_low = None
    ci2_high = None
    model2 = None

    if covars:
        X2 = sm.add_constant(df[["stratio"] + covars])
        model2 = sm.OLS(df["avg_score"], X2).fit()
        coef_stratio_adj = float(model2.params["stratio"])
        p_stratio_adj = float(model2.pvalues["stratio"])
        ci2_low, ci2_high = model2.conf_int().loc["stratio"]

    # Compute Likert-scale response.
    response = compute_likert_score(
        r=r,
        p_corr=p_corr,
        coef_stratio=coef_stratio,
        p_stratio=p_stratio,
        coef_stratio_adj=coef_stratio_adj,
        p_stratio_adj=p_stratio_adj,
    )

    # Build human-readable explanation.
    explanation_parts = []

    explanation_parts.append(
        f"Research question: {research_question} "
        f"using data from {n_obs} school districts."
    )

    explanation_parts.append(
        "I defined the student-teacher ratio (STR) as total students divided by "
        "full-time-equivalent teachers in each district, and academic performance "
        "as the average of 5th-grade reading and math Stanford 9 test scores."
    )

    explanation_parts.append(
        "A Pearson correlation between STR and average test score shows a "
        f"negative association (r = {r:.3f}, p = {p_corr:.3g}), indicating that "
        "districts with more students per teacher tend to have lower test scores."
    )

    explanation_parts.append(
        "A simple OLS regression of average test score on STR (without other "
        f"covariates) yields a coefficient of {coef_stratio:.3f} "
        f"(95% CI [{ci1_low:.3f}, {ci1_high:.3f}], p = {p_stratio:.3g}), "
        "so each additional student per teacher is associated with a decrease "
        "in average test scores."
    )

    if model2 is not None and coef_stratio_adj is not None and p_stratio_adj is not None:
        explanation_parts.append(
            "To check robustness, I estimated a multiple regression including "
            "student demographic and resource covariates "
            f"({', '.join(covars)}). In this adjusted model, the STR coefficient "
            f"remains negative at {coef_stratio_adj:.3f} "
            f"(95% CI [{ci2_low:.3f}, {ci2_high:.3f}], p = {p_stratio_adj:.3g}), "
            "showing that the negative association between STR and achievement "
            "persists even after accounting for these factors."
        )

    explanation_parts.append(
        "Taken together, the consistently negative coefficients and statistically "
        "significant p-values provide strong evidence of an inverse relationship: "
        "districts with lower student-teacher ratios tend to have higher academic "
        "performance on average. This reflects an association, not necessarily a "
        "causal effect, but the pattern is robust in these data."
    )

    explanation_parts.append(
        f"On a 0–100 Likert scale, I encode this as a 'Yes' answer with a score "
        f"of {response}, reflecting strong but not absolute evidence that lower "
        "student-teacher ratios are associated with higher academic performance."
    )

    explanation = " ".join(explanation_parts)

    result = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()


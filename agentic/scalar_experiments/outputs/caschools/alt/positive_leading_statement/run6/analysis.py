import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load metadata and data
    info_path = Path("info.json")
    data_path = Path("caschools.csv")

    info = json.loads(info_path.read_text())
    df = pd.read_csv(data_path)

    # Construct key variables
    df = df.copy()
    # Student-teacher ratio: students per teacher (higher = larger classes)
    df["stratio"] = df["students"] / df["teachers"]
    # Overall academic performance as average of reading and math scores
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Descriptive statistics
    stratio_mean = float(df["stratio"].mean())
    stratio_std = float(df["stratio"].std())
    avg_mean = float(df["avg_score"].mean())
    avg_std = float(df["avg_score"].std())

    # Simple correlation
    corr = float(df["stratio"].corr(df["avg_score"]))

    # Simple linear regression: avg_score ~ stratio
    X_simple = sm.add_constant(df[["stratio"]])
    y = df["avg_score"]
    model_simple = sm.OLS(y, X_simple).fit()
    coef_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression controlling for key covariates
    controls = ["income", "english", "lunch", "calworks", "expenditure"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(y, X_multi).fit()
    coef_multi = float(model_multi.params["stratio"])
    p_multi = float(model_multi.pvalues["stratio"])
    r2_multi = float(model_multi.rsquared)

    # Effect of a 5-student change in ratio
    delta_five = coef_simple * 5.0

    # Determine strength of evidence on a 0-100 scale
    response_score = compute_response_score(corr, coef_simple, p_simple, coef_multi, p_multi)

    # Build textual explanation grounded in the fitted models
    explanation = build_explanation(
        info=info,
        stratio_mean=stratio_mean,
        stratio_std=stratio_std,
        avg_mean=avg_mean,
        avg_std=avg_std,
        corr=corr,
        coef_simple=coef_simple,
        p_simple=p_simple,
        r2_simple=r2_simple,
        coef_multi=coef_multi,
        p_multi=p_multi,
        r2_multi=r2_multi,
        delta_five=delta_five,
        response_score=response_score,
    )

    # Write required JSON output
    conclusion = {"response": int(response_score), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


def compute_response_score(
    corr: float,
    coef_simple: float,
    p_simple: float,
    coef_multi: float,
    p_multi: float,
) -> int:
    """Map statistical evidence to a 0–100 Likert response.

    0 = strong "No" (no evidence of association)
    100 = strong "Yes" (strong, consistent evidence that lower ratios
    are associated with higher performance).
    """

    score = 50.0  # start at neutral

    # Direction of association (we expect corr < 0 and coefficients < 0)
    if corr < 0:
        score += min(10.0, 50.0 * abs(corr))  # up to +10 points
    else:
        score -= min(10.0, 50.0 * abs(corr))  # penalize if positive

    # Simple regression evidence
    if coef_simple < 0 and p_simple < 0.05:
        score += 10.0
        if p_simple < 0.01:
            score += 5.0
        if p_simple < 0.001:
            score += 5.0
    elif coef_simple > 0 and p_simple < 0.05:
        score -= 15.0

    # Multiple regression evidence (more important)
    if coef_multi < 0 and p_multi < 0.05:
        score += 15.0
        if p_multi < 0.01:
            score += 10.0
        if p_multi < 0.001:
            score += 10.0
    elif coef_multi > 0 and p_multi < 0.05:
        score -= 25.0
    else:
        # If the coefficient is not statistically different from zero,
        # slightly downgrade the strength of evidence.
        if p_multi >= 0.05:
            score -= 10.0

    # Clamp to [0, 100] and round to nearest integer
    score = max(0.0, min(100.0, score))
    return int(round(score))


def build_explanation(
    *,
    info: dict,
    stratio_mean: float,
    stratio_std: float,
    avg_mean: float,
    avg_std: float,
    corr: float,
    coef_simple: float,
    p_simple: float,
    r2_simple: float,
    coef_multi: float,
    p_multi: float,
    r2_multi: float,
    delta_five: float,
    response_score: int,
) -> str:
    question = info.get("research_questions", [""])[0]

    direction_phrase = (
        "higher student–teacher ratios (larger classes) are associated with lower test scores"
        if corr < 0
        else "higher student–teacher ratios are associated with higher test scores"
        if corr > 0
        else "there is essentially no linear association between the student–teacher ratio and test scores"
    )

    simple_inference = (
        "The coefficient on the student–teacher ratio was negative and statistically significant"
        if (coef_simple < 0 and p_simple < 0.05)
        else "The coefficient on the student–teacher ratio was not statistically distinguishable from zero"
        if p_simple >= 0.05
        else "The coefficient on the student–teacher ratio was positive and statistically significant"
    )

    multi_inference = (
        "even after adjusting for income, poverty (CalWorks and reduced-price-lunch shares), percent English learners, and per-pupil expenditures"
        if (coef_multi < 0 and p_multi < 0.05)
        else "however, once I adjust for income, poverty, English-learner share, and expenditures, the association with the student–teacher ratio is no longer statistically distinguishable from zero"
        if p_multi >= 0.05
        else "but, after adjusting for income, poverty, English-learner share, and expenditures, the coefficient on the student–teacher ratio becomes positive and statistically significant"
    )

    if coef_multi < 0:
        direction_effect = "lower ratios (smaller classes) are associated with higher academic performance"
    elif coef_multi > 0:
        direction_effect = "lower ratios are associated with lower academic performance"
    else:
        direction_effect = "there is little evidence that the ratio is related to academic performance"

    if response_score >= 70:
        overall_conclusion = (
            "Overall, the evidence supports a 'Yes' answer: the raw data show a clear negative association between student–teacher ratios and test scores, "
            "and although part of this relationship is explained by socioeconomic and demographic differences across districts, the remaining evidence still points toward better performance in districts with lower student–teacher ratios."
        )
    elif response_score <= 30:
        overall_conclusion = (
            "Overall, the evidence supports a 'No' answer: after examining both unadjusted and covariate-adjusted models, there is little consistent statistical evidence that student–teacher ratios are meaningfully related to academic performance in this dataset."
        )
    else:
        overall_conclusion = (
            "Overall, the evidence is mixed: the simple association between student–teacher ratios and test scores is in the expected direction, "
            "but once key covariates are controlled for, the remaining relationship is modest and not always statistically distinguishable from zero, so the data do not provide a strong answer either way."
        )

    explanation = (
        f"Research question: {question}\n\n"
        "Data and variable construction: I used the caschools dataset containing 420 California K-6 and K-8 school districts in 1998–1999. "
        "I defined the student–teacher ratio as total students divided by the number of teachers, so higher values correspond to larger classes. "
        f"The ratio has mean {stratio_mean:.1f} students per teacher (SD {stratio_std:.1f}). "
        f"Academic performance was summarized as the average of district-level reading and math scores, with mean {avg_mean:.1f} (SD {avg_std:.1f}). "
        f"The Pearson correlation between the student–teacher ratio and average test score is {corr:.2f}, meaning that {direction_phrase}. "
        "\n\nStatistical models: I first estimated a simple linear regression of average test score on the student–teacher ratio. "
        f"{simple_inference} (coefficient {coef_simple:.2f} points per additional student per teacher, p = {p_simple:.3g}, R² = {r2_simple:.2f}). "
        f"A 5-student increase in the ratio is associated with about {delta_five:.1f} points change in the average test score in this model. "
        "Next, I estimated a multiple linear regression including controls for district income, the shares of students on CalWorks and reduced-price lunch, the percentage of English learners, and per-pupil expenditure. "
        f"In this model, the coefficient on the student–teacher ratio is {coef_multi:.2f} (p = {p_multi:.3g}, R² = {r2_multi:.2f}), and {multi_inference}. "
        f"Taken together, these models indicate that {direction_effect}. "
        "Because the data are observational and aggregated at the district level, the results describe associations rather than causal effects of class-size reductions. "
        f"\n\nOverall assessment on a 0–100 scale: Based on the direction, magnitude, and statistical significance of the association across models, I assign a response score of {response_score} on the 0–100 Likert scale, where higher values indicate stronger evidence that lower student–teacher ratios are associated with higher academic performance. "
        f"{overall_conclusion}"
    )

    return explanation


if __name__ == "__main__":
    main()

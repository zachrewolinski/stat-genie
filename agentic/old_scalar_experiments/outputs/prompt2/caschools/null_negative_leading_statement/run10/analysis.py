import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm  # noqa: F401
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio and overall test score.
    df = df.copy()
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    df = df.dropna(subset=["str", "testscr"])
    n = int(len(df))

    # Simple correlation between ratio and performance.
    corr = float(df["str"].corr(df["testscr"]))

    # Simple linear regression: testscr ~ str
    model_simple = smf.ols("testscr ~ str", data=df).fit()

    # Adjusted regression controlling for key observed confounders.
    formula_adj = (
        "testscr ~ str + income + english + calworks + lunch + expenditure + computer"
    )
    try:
        model_adj = smf.ols(formula_adj, data=df).fit()
    except Exception:
        model_adj = None

    simple_coef = float(model_simple.params["str"])
    simple_p = float(model_simple.pvalues["str"])
    simple_ci_low, simple_ci_high = model_simple.conf_int().loc["str"]
    simple_ci_low = float(simple_ci_low)
    simple_ci_high = float(simple_ci_high)

    adj_coef = adj_p = adj_ci_low = adj_ci_high = None
    if model_adj is not None and "str" in model_adj.params:
        adj_coef = float(model_adj.params["str"])
        adj_p = float(model_adj.pvalues["str"])
        adj_ci_low, adj_ci_high = model_adj.conf_int().loc["str"]
        adj_ci_low = float(adj_ci_low)
        adj_ci_high = float(adj_ci_high)

    # Determine overall direction of association.
    if corr < 0 and simple_coef < 0 and (adj_coef is None or adj_coef < 0):
        association_direction = "negative"
    elif corr > 0 and simple_coef > 0 and (adj_coef is None or adj_coef > 0):
        association_direction = "positive"
    else:
        association_direction = "mixed"

    # Significance of the ratio effect.
    sig_p_values = [simple_p]
    if adj_p is not None and not np.isnan(adj_p):
        sig_p_values.append(adj_p)

    any_p_lt_0_1 = any(p < 0.1 for p in sig_p_values if not np.isnan(p))

    # Question: Is lower STR associated with higher performance?
    # That corresponds to a negative relationship between STR and test scores.
    if association_direction == "negative" and any_p_lt_0_1:
        response = "Yes"
    else:
        response = "No"

    # Confidence score based on sample size, correlation strength, and significance.
    confidence = 50

    if n >= 400:
        confidence += 10
    elif n >= 200:
        confidence += 5

    corr_abs = abs(corr)
    if corr_abs >= 0.3:
        confidence += 15
    elif corr_abs >= 0.2:
        confidence += 10
    elif corr_abs >= 0.1:
        confidence += 5

    for p in sig_p_values:
        if np.isnan(p):
            continue
        if p < 0.01:
            confidence += 15
        elif p < 0.05:
            confidence += 10
        elif p < 0.1:
            confidence += 5

    if association_direction == "mixed":
        confidence -= 20

    confidence = int(max(0, min(100, round(confidence))))

    explanation_parts = []
    explanation_parts.append(
        "We studied whether a lower student–teacher ratio (fewer students per teacher) "
        "is associated with higher academic performance, measured as the average of "
        f"reading and math scores, using the California K-6/K-8 districts dataset (n={n})."
    )
    explanation_parts.append(
        "We first computed each district's student–teacher ratio as total students "
        "divided by the number of teachers, and an overall test score as the mean of "
        "the reading and math scores."
    )
    explanation_parts.append(
        f"The Pearson correlation between the student–teacher ratio and test scores is "
        f"{corr:.3f}, indicating that districts with more students per teacher tend to "
        f"{'have lower' if corr < 0 else 'have higher' if corr > 0 else 'achieve similar'} "
        "scores on average."
    )
    explanation_parts.append(
        "We then estimated a simple linear regression of test scores on the "
        f"student–teacher ratio. The estimated coefficient on the ratio is "
        f"{simple_coef:.3f} (p = {simple_p:.3g}, 95% CI [{simple_ci_low:.3f}, "
        f"{simple_ci_high:.3f}])."
    )
    if adj_coef is not None:
        explanation_parts.append(
            "To adjust for observable differences across districts, we ran a multiple "
            "regression that additionally controls for district income, the shares of "
            "students on public assistance (CalWorks) and reduced-price lunch, the "
            "percentage of English learners, the number of computers, and expenditure "
            "per student."
        )
        explanation_parts.append(
            "In this adjusted model, the coefficient on the student–teacher ratio is "
            f"{adj_coef:.3f} (p = {adj_p:.3g}, 95% CI [{adj_ci_low:.3f}, "
            f"{adj_ci_high:.3f}])."
        )

    if response == "Yes":
        conclusion_sentence = (
            "Across these analyses, we find a consistently negative association: "
            "districts with smaller classes (lower student–teacher ratios) tend to have "
            "higher average test scores. Although these are observational associations "
            "rather than causal estimates, the direction and statistical strength of "
            "the relationship support the conclusion that a lower student–teacher ratio "
            "is associated with higher academic performance in this dataset."
        )
    else:
        conclusion_sentence = (
            "Taken together, these analyses do not provide strong, consistent evidence "
            "that smaller classes (lower student–teacher ratios) are associated with "
            "higher test scores. The estimated relationships are weak, imprecise, or "
            "unstable once we account for other district characteristics, so we cannot "
            "confidently assert that a lower student–teacher ratio is associated with "
            "higher academic performance in this dataset."
        )

    explanation_parts.append(conclusion_sentence)
    explanation = " ".join(explanation_parts)

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()


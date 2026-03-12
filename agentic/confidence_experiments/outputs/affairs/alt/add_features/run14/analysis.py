import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_likert_score(coef_children: float, p_children: float, delta_prob: float) -> int:
    """Map direction, significance, and effect size to a 0–100 Likert score."""
    # Base score from direction and significance
    if coef_children < 0:
        # Direction consistent with "children decrease affairs"
        if p_children < 0.001:
            score = 95
        elif p_children < 0.01:
            score = 90
        elif p_children < 0.05:
            score = 80
        elif p_children < 0.1:
            score = 65
        else:
            # Direction is favorable but not statistically supported
            score = 40
    else:
        # Direction inconsistent with the research question or essentially null
        if p_children < 0.001:
            score = 5
        elif p_children < 0.01:
            score = 10
        elif p_children < 0.05:
            score = 15
        elif p_children < 0.1:
            score = 25
        else:
            # No clear evidence either way
            score = 35

    # Adjust for magnitude of effect in predicted probabilities
    abs_delta = abs(delta_prob)
    if abs_delta >= 0.15:
        score += 10
    elif abs_delta >= 0.05:
        score += 5
    elif abs_delta <= 0.01:
        score -= 5

    score = max(0, min(100, int(round(score))))
    return score


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "affairs.csv"

    df = pd.read_csv(data_path)

    # Focus on whether there was any affair in the past year.
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Restrict to observations with clear children information.
    df = df[df["children"].isin(["yes", "no"])].copy()
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Basic group-wise incidence of affairs.
    rates = df.groupby("children")["had_affair"].agg(["mean", "count"])

    # Encode gender as binary (male vs female).
    df["gender_male"] = (df["gender"] == "male").astype(int)

    # Select covariates: age, years married, religiousness, rating, gender.
    predictors = ["children_yes", "age", "yearsmarried", "religiousness", "rating", "gender_male"]
    model_data = df[["had_affair"] + predictors].replace([np.inf, -np.inf], np.nan).dropna()

    y = model_data["had_affair"]
    X = sm.add_constant(model_data[predictors], has_constant="add")

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    coef_children = float(result.params["children_yes"])
    p_children = float(result.pvalues["children_yes"])
    se_children = float(result.bse["children_yes"])

    # Predicted probabilities at average covariate values with and without children.
    mean_vals = X.mean()
    mean_no_child = mean_vals.copy()
    mean_no_child["children_yes"] = 0
    mean_child = mean_vals.copy()
    mean_child["children_yes"] = 1

    prob_no_child = float(result.predict(mean_no_child))
    prob_child = float(result.predict(mean_child))

    # Difference in predicted probabilities in direction of the research question
    # (how much lower is the probability when there are children).
    delta_prob = prob_no_child - prob_child

    # Observed rates by group.
    rate_no_child = float(rates.loc["no", "mean"])
    rate_child = float(rates.loc["yes", "mean"])
    n_no_child = int(rates.loc["no", "count"])
    n_child = int(rates.loc["yes", "count"])

    likert_score = compute_likert_score(coef_children, p_children, delta_prob)
    answer_label = "Yes" if likert_score >= 60 else "No"

    # Qualitative description of evidence strength.
    if p_children < 0.01:
        evidence_strength = "strong"
    elif p_children < 0.05:
        evidence_strength = "moderate"
    elif p_children < 0.1:
        evidence_strength = "weak"
    else:
        evidence_strength = "little or no"

    if p_children >= 0.1:
        evidence_phrase = f"{evidence_strength} statistical evidence that having children meaningfully changes the likelihood of an affair"
    elif coef_children < 0:
        evidence_phrase = f"{evidence_strength} statistical evidence that having children is associated with fewer extramarital affairs"
    else:
        evidence_phrase = f"{evidence_strength} statistical evidence that having children is associated with more extramarital affairs"

    if p_children >= 0.1:
        effect_summary = "does not show a clear systematic relationship with the likelihood of having an affair in this sample"
    elif coef_children < 0 and delta_prob > 0:
        effect_summary = "is associated with a lower predicted probability of having an affair"
    elif coef_children < 0:
        effect_summary = "is associated with slightly lower log-odds of having an affair, though the change in predicted probability is small"
    elif delta_prob < 0:
        effect_summary = "is associated with a higher predicted probability of having an affair"
    else:
        effect_summary = "shows only a very small change in the predicted probability of having an affair"

    n_total = int(len(df))

    explanation = (
        f"In a sample of {n_total} first-marriage respondents from the Fair (1978) extramarital affairs dataset, "
        f"{rate_no_child * 100:.1f}% of those without children (n = {n_no_child}) reported at least one extramarital affair in the past year, "
        f"compared with {rate_child * 100:.1f}% of those with children (n = {n_child}). "
        f"A logistic regression of having any affair on an indicator for having children, controlling for age, years married, religiousness, self-rated marital happiness, and gender, "
        f"estimated a coefficient of {coef_children:.3f} (standard error {se_children:.3f}, p = {p_children:.3f}) for the 'has children' indicator, "
        f"corresponding to predicted probabilities of {prob_no_child * 100:.1f}% without children and {prob_child * 100:.1f}% with children at average covariate values. "
        f"Overall, this provides {evidence_phrase} and suggests that having children {effect_summary}. "
        f"Accordingly, I answer '{answer_label}' to the question 'Does having children decrease engagement in extramarital affairs?', "
        f"with a Likert-scale confidence of {likert_score}/100 that the presence of children reduces such engagement."
    )

    conclusion = {"response": int(likert_score), "explanation": explanation}

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


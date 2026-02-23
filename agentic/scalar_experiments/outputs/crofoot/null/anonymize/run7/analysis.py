import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won
    y = df["feature4"]

    # Relative group size: focal size minus other size
    df["group_size_diff"] = df["feature7"] - df["feature8"]

    # Contest location advantage: other distance to its center minus focal distance to its center.
    # Positive values mean the focal group is closer to its home-range center than the other group.
    df["home_advantage"] = df["feature6"] - df["feature5"]

    X = df[["group_size_diff", "home_advantage"]]
    X = sm.add_constant(X)

    # Fit logistic regression
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    params = result.params
    pvalues = result.pvalues

    coef_gs = params["group_size_diff"]
    coef_home = params["home_advantage"]
    p_gs = float(pvalues["group_size_diff"])
    p_home = float(pvalues["home_advantage"])

    # Odds ratios (effect on odds of focal group winning)
    or_gs = float(np.exp(coef_gs))
    # For home advantage, scale per 100 m for interpretability
    or_home_100m = float(np.exp(coef_home * 100.0))

    # Compute simple Likert-style score from p-values and effect directions.
    # We treat evidence that larger relative group size and being closer to the home-range center
    # increase win probability as supporting a "Yes" answer.
    def evidence_score(p: float) -> float:
        if p < 0.001:
            return 1.0
        if p < 0.01:
            return 0.85
        if p < 0.05:
            return 0.7
        if p < 0.1:
            return 0.5
        if p < 0.2:
            return 0.35
        return 0.2

    score_gs = evidence_score(p_gs) if coef_gs != 0 else 0.2
    score_home = evidence_score(p_home) if coef_home != 0 else 0.2

    # If effects run counter to the intuitive direction, down-weight their contribution.
    if coef_gs < 0:
        score_gs *= 0.5
    if coef_home < 0:
        score_home *= 0.5

    # Combine the two sources of evidence (equal weight) and map to 0-100.
    combined_score = (score_gs + score_home) / 2.0
    response_scalar = int(round(combined_score * 100))

    # Clamp to [0, 100]
    response_scalar = max(0, min(100, response_scalar))

    # Build explanation string summarizing the analysis and key statistics.
    explanation_lines = []
    explanation_lines.append(
        "I modeled the probability that the focal capuchin monkey group won an intergroup contest "
        "using logistic regression with 58 contests."
    )
    explanation_lines.append(
        "The binary outcome was whether the focal group won (feature4), "
        "and the predictors were relative group size (focal group size minus other group size; feature7-feature8) "
        "and a contest location advantage metric (other group distance from its home-range center minus focal group distance; feature6-feature5)."
    )
    explanation_lines.append(
        f"The coefficient for relative group size was {coef_gs:.3f}, giving an odds ratio of {or_gs:.2f} "
        f"with p-value {p_gs:.3g}, so contests where the focal group is larger tend to have higher odds of winning when this effect is statistically supported."
    )
    explanation_lines.append(
        f"The coefficient for contest location advantage was {coef_home:.5f} per meter "
        f"(odds ratio {or_home_100m:.2f} per 100 m advantage; p-value {p_home:.3g}), "
        "indicating how much being closer to its own home-range center affects the focal group's win probability."
    )
    explanation_lines.append(
        "Based on the joint evidence from these coefficients and their statistical significance levels, "
        "I translated the strength of support for an effect of both relative group size and contest location "
        f"on win probability into a 0–100 Likert scale, where higher values represent stronger evidence for a 'Yes' answer. "
        f"The resulting score was {response_scalar}."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": response_scalar,
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()


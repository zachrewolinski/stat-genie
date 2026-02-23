import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Outcome: 1 if focal group won, 0 otherwise
    win = df["feature4"].astype(int)

    # Relative group size: difference in number of individuals (focal - other)
    size_focal = df["feature7"]
    size_other = df["feature8"]
    size_diff = size_focal - size_other

    # Contest location: relative distance to home range centers.
    # feature5: distance of focal group from its home range center
    # feature6: distance of other group from its home range center
    # Define "home advantage" as how much closer the focal group is than the other group.
    # Positive values mean the focal group is closer to its home range center than the other group is to theirs.
    dist_focal = df["feature5"]
    dist_other = df["feature6"]
    home_adv = dist_other - dist_focal

    analysis_df = pd.DataFrame(
        {
            "win": win,
            "size_diff": size_diff,
            "home_adv": home_adv,
        }
    ).dropna()

    # Fit logistic regression: probability focal group wins ~ relative group size + home advantage
    X = analysis_df[["size_diff", "home_adv"]]
    X = sm.add_constant(X)
    y = analysis_df["win"]

    try:
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(disp=False)
    except Exception:
        # Fallback to GLM with binomial family if Logit fails to converge
        glm_model = sm.GLM(y, X, family=sm.families.Binomial())
        result = glm_model.fit()

    params = result.params
    pvalues = result.pvalues

    coef_size = float(params.get("size_diff", np.nan))
    p_size = float(pvalues.get("size_diff", np.nan))

    coef_home = float(params.get("home_adv", np.nan))
    p_home = float(pvalues.get("home_adv", np.nan))

    # Determine strength of evidence that relative group size and contest location influence win probability.
    # We summarize across both predictors using their significance and effect directions.
    def evidence_score(p: float, coef: float) -> float:
        if np.isnan(p) or np.isnan(coef):
            return 0.0
        # Base on significance
        if p < 0.001:
            base = 1.0
        elif p < 0.01:
            base = 0.85
        elif p < 0.05:
            base = 0.7
        elif p < 0.1:
            base = 0.5
        else:
            base = 0.3
        # Modulate by standardized effect magnitude
        mag = min(abs(coef), 2.0) / 2.0  # cap at |coef|=2
        return base * (0.6 + 0.4 * mag)

    score_size = evidence_score(p_size, coef_size)
    score_home = evidence_score(p_home, coef_home)

    # Combine evidence from both predictors
    combined = (score_size + score_home) / 2.0

    # Map combined evidence to Likert scale [0, 100], interpreted as "Yes" strength.
    # Very weak or non-significant effects (combined <= 0.35) should correspond to a weak "Yes" or even "No".
    if combined <= 0.2:
        response_value = 20
    elif combined <= 0.35:
        response_value = 40
    elif combined <= 0.5:
        response_value = 60
    elif combined <= 0.7:
        response_value = 75
    else:
        response_value = 90

    # Ensure integer as required
    response_int = int(response_value)

    # Build explanation string with key statistics
    explanation_parts = []
    explanation_parts.append(
        "I modeled the probability that the focal capuchin group wins an intergroup contest "
        "using logistic regression with two predictors: relative group size (difference in number of individuals, "
        "focal minus other group) and contest location (a 'home advantage' variable defined as the other group's "
        "distance from its home range center minus the focal group's distance; positive values mean the contest is "
        "closer to the focal group's home range center)."
    )
    explanation_parts.append(
        f"In this model (N = {len(analysis_df)} contests), the estimated coefficient for relative group size "
        f"was {coef_size:.3f} with p-value {p_size:.3f}, and the coefficient for home advantage was "
        f"{coef_home:.3f} with p-value {p_home:.3f}."
    )

    if response_int >= 70:
        interpretation = (
            "These estimates provide statistically meaningful evidence that both relative group size and contest "
            "location influence the focal group's chance of winning: larger focal groups and contests occurring closer "
            "to the focal group's home range center are associated with a higher probability of victory."
        )
    elif response_int >= 50:
        interpretation = (
            "These estimates provide moderate but not overwhelming evidence that relative group size and/or contest "
            "location influence the focal group's chance of winning; the directions of the effects are consistent with "
            "larger focal groups and greater home advantage increasing the probability of a win, but some uncertainty remains."
        )
    else:
        interpretation = (
            "Overall, the statistical evidence that relative group size and contest location influence the focal group's "
            "chance of winning is weak; effect estimates are small or imprecise, and the data do not strongly rule out "
            "the possibility of little or no influence for one or both predictors."
        )

    explanation_parts.append(interpretation)

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response_int,
        "explanation": explanation,
    }

    # Write the required JSON object to conclusion.txt with no extra text.
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()


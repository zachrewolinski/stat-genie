import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load metadata and data
    info = json.loads(Path("info.json").read_text())
    df = pd.read_csv("crofoot.csv")

    # Construct key predictors according to research question
    # Relative group size: focal minus other (positive if focal larger)
    df["rel_n"] = df["n_focal"] - df["n_other"]

    # Contest location: relative centrality (other distance minus focal distance)
    # Positive values: contest closer to focal home-range center
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]

    # Standardise predictors for interpretability
    for col in ["rel_n", "rel_dist"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0:
            df[f"z_{col}"] = 0.0
        else:
            df[f"z_{col}"] = (df[col] - mean) / std

    y = df["win"]
    X = df[["z_rel_n", "z_rel_dist"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    try:
        result = logit_model.fit(disp=0)
    except Exception as exc:  # pragma: no cover - defensive
        explanation = (
            "Failed to fit logistic regression model due to: "
            f"{exc}. Cannot robustly assess effects of relative "
            "group size and contest location on win probability."
        )
        output = {"response": 50, "explanation": explanation}
        Path("conclusion.txt").write_text(json.dumps(output))
        return

    # Collect coefficient estimates and p-values
    params = result.params
    pvalues = result.pvalues

    coef_rel_n = float(params.get("z_rel_n", np.nan))
    p_rel_n = float(pvalues.get("z_rel_n", np.nan))
    coef_rel_dist = float(params.get("z_rel_dist", np.nan))
    p_rel_dist = float(pvalues.get("z_rel_dist", np.nan))

    # Determine strength of evidence
    sig_rel_n = p_rel_n < 0.05
    sig_rel_dist = p_rel_dist < 0.05

    # Map statistical evidence to Likert-style 0-100 response
    # Start from neutral 50 and adjust
    score = 50

    # Effect of relative group size
    if sig_rel_n and coef_rel_n > 0:
        score += 20
    elif sig_rel_n and coef_rel_n < 0:
        score -= 10

    # Effect of contest location (closer to focal center)
    if sig_rel_dist and coef_rel_dist > 0:
        score += 20
    elif sig_rel_dist and coef_rel_dist < 0:
        score -= 10

    # Cap within [0, 100] and convert to int
    score = int(min(max(score, 0), 100))

    # Compose explanation
    lines = []
    rq = info.get("research_questions", [""])[0]
    lines.append(
        "Research question: "
        "Do relative group size and contest location influence the probability "
        "of a capuchin group winning an intergroup contest?"
    )
    lines.append(
        "I addressed this using a logistic regression with win (1 = focal group "
        "won, 0 = other group won) as the outcome and two predictors: (i) "
        "relative group size (focal minus other group size) and (ii) relative "
        "contest location (other group distance from its home-range centre minus "
        "focal group distance)."
    )

    lines.append(
        "Both predictors were standardised, and I estimated a logistic regression "
        f"on {len(df)} contests. The coefficient for relative group size was "
        f"{coef_rel_n:.3f} with p-value {p_rel_n:.3f}, and the coefficient for "
        f"relative contest location was {coef_rel_dist:.3f} with p-value "
        f"{p_rel_dist:.3f}."
    )

    if sig_rel_n or sig_rel_dist:
        direction_parts = []
        if sig_rel_n:
            if coef_rel_n > 0:
                direction_parts.append(
                    "larger focal groups had a significantly higher probability "
                    "of winning than smaller focal groups"
                )
            else:
                direction_parts.append(
                    "larger focal groups had a significantly lower probability "
                    "of winning than smaller focal groups"
                )
        if sig_rel_dist:
            if coef_rel_dist > 0:
                direction_parts.append(
                    "contests occurring closer to the focal group's home-range "
                    "centre were significantly more likely to be won by the focal group"
                )
            else:
                direction_parts.append(
                    "contests occurring closer to the other group's home-range "
                    "centre were significantly more likely to be won by the other group"
                )

        lines.append(
            "Taken together, these results provide statistically significant "
            "evidence that " + " and ".join(direction_parts) + ", indicating that "
            "both relative group size and contest location influence contest "
            "outcomes in this dataset."
        )
    else:
        lines.append(
            "Neither relative group size nor contest location achieved "
            "conventional statistical significance (p < 0.05), so there is "
            "insufficient evidence in this dataset to claim that they affect "
            "the probability of winning."
        )

    if score >= 60:
        lines.append(
            f"Based on these analyses, I answer the research question as a "
            f"'Yes' with confidence corresponding to {score} on a 0–100 scale, "
            "where higher values indicate stronger evidence that relative group "
            "size and contest location influence win probability."
        )
    elif score <= 40:
        lines.append(
            f"Based on these analyses, I answer the research question as a "
            f"'No' with confidence corresponding to {score} on a 0–100 scale, "
            "reflecting limited evidence that these variables influence win "
            "probability in this sample."
        )
    else:
        lines.append(
            f"Based on these analyses, the evidence is equivocal; I give an "
            f"intermediate score of {score} on a 0–100 scale, reflecting "
            "uncertain but suggestive effects of relative group size and contest "
            "location on win probability."
        )

    explanation = " ".join(lines)

    output = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output))


if __name__ == "__main__":
    main()

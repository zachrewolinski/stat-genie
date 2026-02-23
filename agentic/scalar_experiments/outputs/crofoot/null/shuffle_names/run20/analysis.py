import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).parent
    data_path = base_dir / "crofoot.csv"
    info_path = base_dir / "info.json"
    conclusion_path = base_dir / "conclusion.txt"

    # Load metadata (not strictly required for the analysis logic, but used for context)
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    # Load dataset
    df = pd.read_csv(data_path)

    # According to info.json metadata:
    # - m_focal: 1 if focal won, 0 if other won  -> outcome
    # - f_other: number of individuals in focal group
    # - win: number of individuals in other group
    # - m_other: distance (m) of focal group from its home-range center
    # - n_focal: distance (m) of other group from its home-range center

    win_indicator = df["m_focal"].astype(int)
    size_focal = df["f_other"].astype(float)
    size_other = df["win"].astype(float)
    dist_focal = df["m_other"].astype(float)
    dist_other = df["n_focal"].astype(float)

    # Relative group size and contest location metrics
    # Positive rel_size means focal group is larger.
    rel_size = size_focal - size_other
    # Negative rel_dist means contest is closer to focal group's home-range center.
    rel_dist = dist_focal - dist_other

    # Standardize predictors for numerical stability and to make coefficients comparable.
    def zscore(x: pd.Series) -> pd.Series:
        return (x - x.mean()) / x.std(ddof=0)

    X = pd.DataFrame(
        {
            "rel_size_z": zscore(rel_size),
            "rel_dist_z": zscore(rel_dist),
        }
    )
    X = sm.add_constant(X)

    # Fit logistic regression: P(focal wins) ~ relative group size + relative distance
    logit_model = sm.Logit(win_indicator, X)
    result = logit_model.fit(disp=False)

    params = result.params
    pvalues = result.pvalues

    # Compute odds ratios for interpretability
    odds_ratios = np.exp(params)

    # Extract statistics of interest
    coef_size = params["rel_size_z"]
    p_size = pvalues["rel_size_z"]
    or_size = odds_ratios["rel_size_z"]

    coef_dist = params["rel_dist_z"]
    p_dist = pvalues["rel_dist_z"]
    or_dist = odds_ratios["rel_dist_z"]

    # Simple heuristic to map evidence to a 0-100 scale.
    # We consider p <= 0.05 as conventionally statistically significant,
    # p between 0.05 and 0.10 as suggestive, and > 0.10 as weak/not supported.
    def evidence_score(p: float, or_value: float) -> float:
        if p <= 0.01:
            base = 90.0
        elif p <= 0.05:
            base = 75.0
        elif p <= 0.10:
            base = 60.0
        elif p <= 0.20:
            base = 45.0
        else:
            base = 25.0

        # Adjust for effect size magnitude (odds ratio deviation from 1).
        effect_strength = abs(np.log(or_value))
        if effect_strength > 0.7:  # roughly OR < 0.5 or > 2.0
            base += 5.0
        elif effect_strength < 0.2:
            base -= 5.0

        return base

    size_score = evidence_score(p_size, or_size)
    dist_score = evidence_score(p_dist, or_dist)

    # Overall assessment: average evidence for the two predictors.
    # If both clearly non-significant, this average will be low.
    overall_score = float((size_score + dist_score) / 2.0)

    # Clip to [0, 100] and convert to integer as required.
    response_int = int(round(np.clip(overall_score, 0.0, 100.0)))

    yes_no = "Yes" if response_int >= 50 else "No"

    # Build explanation string, including key statistics.
    explanation_lines = []
    explanation_lines.append(
        f"Research question: {research_question}"
    )
    explanation_lines.append(
        "I modeled the probability that the focal capuchin group won an intergroup contest "
        "using a logistic regression with two standardized predictors: (1) relative group size "
        "(focal group size minus other group size) and (2) relative contest location "
        "(focal group distance from its home-range center minus the other group’s distance)."
    )
    explanation_lines.append(
        f"For relative group size, the standardized coefficient was {coef_size:.3f}, "
        f"odds ratio {or_size:.3f}, p-value {p_size:.3f}."
    )
    explanation_lines.append(
        f"For relative contest location, the standardized coefficient was {coef_dist:.3f}, "
        f"odds ratio {or_dist:.3f}, p-value {p_dist:.3f}."
    )

    if response_int >= 50:
        conclusion_text = (
            f"{yes_no}: There is statistical evidence that one or both of relative group size "
            f"and contest location influence the probability that a focal group wins a contest "
            f"(overall support score {response_int} on a 0–100 scale)."
        )
    else:
        conclusion_text = (
            f"{yes_no}: The data do not provide strong statistical evidence that relative group size "
            f"or contest location meaningfully influence the probability that a focal group wins a contest "
            f"(overall support score {response_int} on a 0–100 scale)."
        )

    explanation_lines.append(conclusion_text)

    explanation = " ".join(explanation_lines)

    conclusion_obj = {
        "response": response_int,
        "explanation": explanation,
    }

    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion_obj, f)


if __name__ == "__main__":
    main()


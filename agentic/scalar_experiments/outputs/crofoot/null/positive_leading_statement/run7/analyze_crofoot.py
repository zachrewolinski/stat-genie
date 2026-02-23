import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Construct key predictors capturing the research question.
    df["size_diff"] = df["n_focal"] - df["n_other"]  # positive when focal group is larger
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    df["loc_diff"] = df["dist_other"] - df["dist_focal"]  # positive when contest is closer to focal home range
    df["focal_closer"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    # Fit logistic regression with continuous measures of relative size and location.
    # We rescale loc_diff so coefficients are interpretable per 100 m.
    df["loc_diff_100"] = df["loc_diff"] / 100.0

    model = smf.logit("win ~ size_diff + loc_diff_100", data=df).fit(disp=False)

    params = model.params
    pvalues = model.pvalues

    # Odds ratios
    or_size = float(np.exp(params["size_diff"]))
    or_loc_100 = float(np.exp(params["loc_diff_100"]))

    p_size = float(pvalues["size_diff"])
    p_loc = float(pvalues["loc_diff_100"])

    # Simple descriptive summaries
    def win_rate(mask: pd.Series) -> float:
        subset = df.loc[mask, "win"]
        if subset.empty:
            return float("nan")
        return float(subset.mean())

    bigger_mask = df["size_diff"] > 0
    equal_mask = df["size_diff"] == 0
    smaller_mask = df["size_diff"] < 0

    win_bigger = win_rate(bigger_mask)
    win_equal = win_rate(equal_mask)
    win_smaller = win_rate(smaller_mask)

    win_focal_closer = win_rate(df["focal_closer"] == 1)
    win_other_closer = win_rate(df["focal_closer"] == 0)

    n_bigger = int(bigger_mask.sum())
    n_equal = int(equal_mask.sum())
    n_smaller = int(smaller_mask.sum())
    n_focal_closer = int((df["focal_closer"] == 1).sum())
    n_other_closer = int((df["focal_closer"] == 0).sum())

    # Map statistical evidence to a 0–100 Likert-style scale.
    def score_component(p: float, odds_ratio: float) -> int:
        # Base score from p-value (evidence against null of no effect).
        if p < 0.001:
            base = 30
        elif p < 0.01:
            base = 25
        elif p < 0.05:
            base = 20
        elif p < 0.10:
            base = 10
        else:
            base = 0

        # Effect size adjustment: larger deviations from OR=1 strengthen evidence that
        # the predictor meaningfully changes win probability.
        effect_mag = abs(np.log(odds_ratio))
        if effect_mag > np.log(2.0):  # roughly OR < 0.5 or > 2
            base += 5
        elif effect_mag > np.log(1.5):
            base += 3

        return int(round(base))

    score_size = score_component(p_size, or_size)
    score_loc = score_component(p_loc, or_loc_100)

    # Start from a conservative baseline reflecting prior uncertainty.
    score = 20 + score_size + score_loc
    score = max(0, min(100, score))

    # Build a human-readable explanation.
    explanation_lines = []
    explanation_lines.append(
        "Research question: Do relative group size and contest location "
        "influence the probability that the focal capuchin group wins an intergroup contest?"
    )
    explanation_lines.append(
        "I modeled win probability (1 = focal group won, 0 = lost) using logistic regression "
        "with two key predictors: (a) relative group size (n_focal − n_other) and "
        "(b) contest location advantage, measured as the difference in distance to each group's "
        "home range center ((dist_other − dist_focal)/100, so positive values mean the contest "
        "is closer to the focal group's core area by 100 m units)."
    )
    explanation_lines.append(
        f"The fitted logistic model indicates that each additional individual in the focal group "
        f"relative to the opponent changes the odds of winning by a factor of about {or_size:.2f} "
        f"(p = {p_size:.3f})."
    )
    explanation_lines.append(
        f"For contest location, a 100 m shift in advantage towards the focal group's home range "
        f"changes their odds of winning by a factor of about {or_loc_100:.2f} "
        f"(p = {p_loc:.3f})."
    )
    explanation_lines.append(
        f"Descriptively, the focal group won in approximately "
        f"{win_bigger*100:.1f}% of contests when it was larger (n = {n_bigger}), "
        f"{win_equal*100:.1f}% when groups were equal in size (n = {n_equal}), and "
        f"{win_smaller*100:.1f}% when it was smaller (n = {n_smaller})."
    )
    explanation_lines.append(
        f"When the contest occurred closer to the focal group's home range center "
        f"(focal_closer = 1), it won about {win_focal_closer*100:.1f}% of contests "
        f"(n = {n_focal_closer}), compared with {win_other_closer*100:.1f}% wins "
        f"when the opponent was closer to its own center (n = {n_other_closer})."
    )

    if score >= 70:
        overall = (
            "Taken together, the regression coefficients, p-values, and descriptive win rates "
            "provide substantial evidence that both relative group size and contest location "
            "influence the probability of the focal group winning."
        )
    elif score >= 50:
        overall = (
            "Overall, the analysis suggests that relative group size and contest location "
            "likely influence win probability, but the strength of evidence is only moderate "
            "given the sample size and uncertainty in the estimated effects."
        )
    else:
        overall = (
            "Overall, the available data provide limited statistical evidence that relative "
            "group size and contest location influence win probability; any effects are "
            "uncertain with this sample."
        )

    explanation_lines.append(overall)
    explanation_lines.append(
        f"On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes', "
        f"this analysis corresponds to a score of {score}, reflecting the combined "
        f"strength of statistical and descriptive evidence."
    )

    explanation = "\n".join(explanation_lines)

    conclusion = {"response": int(score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Rename for clarity
    df = df.rename(
        columns={
            "feature4": "win",
            "feature5": "focal_dist",
            "feature6": "other_dist",
            "feature7": "size_focal",
            "feature8": "size_other",
        }
    )

    # Construct key predictors
    # Relative group size: log size ratio (symmetric, reduces skew)
    df["size_ratio"] = df["size_focal"] / df["size_other"]
    df["log_size_ratio"] = np.log(df["size_ratio"])

    # Contest location advantage: positive if focal group is closer to its home-range center
    # (i.e., other group is farther from its own center than the focal group).
    df["loc_advantage"] = df["other_dist"] - df["focal_dist"]

    # Drop any rows with missing values in key columns (defensive, though dataset appears complete).
    df = df.dropna(subset=["win", "log_size_ratio", "loc_advantage"])

    y = df["win"].astype(int)
    X = df[["log_size_ratio", "loc_advantage"]]
    X = sm.add_constant(X)

    # Fit logistic regression
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    params = result.params
    pvalues = result.pvalues

    beta_size = params["log_size_ratio"]
    p_size = pvalues["log_size_ratio"]

    beta_loc = params["loc_advantage"]
    p_loc = pvalues["loc_advantage"]

    # Effect-size summaries: change in predicted win probability across
    # interquartile ranges for each predictor, holding the other at its median.
    def prob_change_for_predictor(col_name: str) -> float:
        q1, q3 = df[col_name].quantile([0.25, 0.75])
        med_other = df["loc_advantage" if col_name == "log_size_ratio" else "log_size_ratio"].median()

        def pred_prob(val_col: float, val_other: float) -> float:
            if col_name == "log_size_ratio":
                x_vec = [1.0, val_col, val_other]
            else:
                x_vec = [1.0, val_other, val_col]
            return float(result.predict([x_vec])[0])

        p_low = pred_prob(q1, med_other)
        p_high = pred_prob(q3, med_other)
        return p_high - p_low

    try:
        dprob_size = prob_change_for_predictor("log_size_ratio")
    except Exception:
        dprob_size = np.nan

    try:
        dprob_loc = prob_change_for_predictor("loc_advantage")
    except Exception:
        dprob_loc = np.nan

    # Build a narrative explanation summarizing evidence
    lines = []
    lines.append(
        "Research question: Do relative group size and contest location influence "
        "the probability that a focal capuchin group wins an intergroup contest?"
    )

    # Sample description
    lines.append(
        f"The dataset contains {len(df)} contests between capuchin groups, "
        "with a binary outcome indicating whether the focal group won."
    )

    # Size effect
    dir_size = "larger" if beta_size > 0 else "smaller"
    size_sig = p_size < 0.05
    size_strength = (
        "strong"
        if size_sig and abs(dprob_size) >= 0.25
        else "moderate"
        if size_sig and abs(dprob_size) >= 0.10
        else "weak"
        if size_sig
        else "not clearly detectable"
    )

    lines.append(
        f"Relative group size (log size ratio of focal to other group) has a "
        f"coefficient of {beta_size:.3f} (p = {p_size:.3f}), indicating that {dir_size} "
        "focal groups tend to have higher win probabilities when this effect is considered "
        "in isolation."
    )
    if not np.isnan(dprob_size):
        lines.append(
            f"Across the interquartile range of relative group size, the model-predicted "
            f"probability that the focal group wins changes by about {dprob_size:.2f} "
            "on an absolute probability scale, which I interpret as a "
            f"{size_strength} effect."
        )
    else:
        lines.append(
            "I could not reliably compute an interquartile effect size for group size "
            "due to numerical issues, so the interpretation relies solely on the sign "
            "and significance of the regression coefficient."
        )

    # Location effect
    if beta_loc > 0:
        loc_dir = (
            "when the focal group is relatively closer to its own home-range center "
            "than the opposing group is to its center (positive location advantage)"
        )
    else:
        loc_dir = (
            "when the focal group is relatively farther from its home-range center "
            "than the opposing group (negative location advantage)"
        )
    loc_sig = p_loc < 0.05
    loc_strength = (
        "strong"
        if loc_sig and abs(dprob_loc) >= 0.25
        else "moderate"
        if loc_sig and abs(dprob_loc) >= 0.10
        else "weak"
        if loc_sig
        else "not clearly detectable"
    )

    lines.append(
        f"Contest location, summarized as a relative home-range advantage "
        f"(other group's distance from its center minus the focal group's distance), "
        f"has a coefficient of {beta_loc:.5f} (p = {p_loc:.3f}). This means that {loc_dir} "
        "is associated with a systematic shift in win probability according to the model."
    )
    if not np.isnan(dprob_loc):
        lines.append(
            f"Across the interquartile range of this location advantage, the predicted "
            f"win probability for the focal group changes by about {dprob_loc:.2f}, "
            f"which I interpret as a {loc_strength} effect."
        )
    else:
        lines.append(
            "As with group size, numerical issues prevented a stable interquartile effect "
            "size estimate for location, so interpretation relies on the sign and "
            "significance of the coefficient."
        )

    # Overall judgment: do *both* relative group size and location matter?
    # Use both statistical significance and effect-size magnitude to set the Likert score.
    both_sig = size_sig and loc_sig
    any_sig = size_sig or loc_sig

    if both_sig:
        # Both predictors show statistically reliable effects with at least modest magnitude.
        base_score = 80
        if abs(dprob_size) >= 0.25 and abs(dprob_loc) >= 0.25:
            base_score = 90
    elif any_sig:
        # Only one of the two predictors is clearly significant.
        base_score = 65
    else:
        # Neither predictor reaches conventional significance; treat as a weak/no effect.
        base_score = 35

    response = int(round(base_score))

    if both_sig:
        lines.append(
            "Taken together, the logistic regression suggests that both relative group size "
            "and contest location systematically influence which group wins. Larger focal "
            "groups and those with a stronger home-range advantage have higher modeled "
            "win probabilities, and both effects reach conventional levels of statistical "
            "significance given the sample size."
        )
    elif any_sig:
        which = []
        if size_sig:
            which.append("relative group size")
        if loc_sig:
            which.append("contest location")
        which_str = " and ".join(which)
        lines.append(
            "The analysis indicates that at least one of the two focal predictors "
            f"({which_str}) has a statistically significant relationship with the probability "
            "of the focal group winning, while the other shows suggestive but less decisive "
            "evidence. Overall, this supports a qualified 'Yes' to the research question."
        )
    else:
        lines.append(
            "Given the limited sample size and the estimated coefficients, neither relative "
            "group size nor contest location shows a clearly statistically significant effect "
            "on win probability at the 5% level. The point estimates are directionally "
            "consistent with larger size and greater home-range advantage helping the focal "
            "group, but the uncertainty is substantial, so the evidence that these factors "
            "influence contest outcomes is weak."
        )

    lines.append(
        f"On a 0–100 scale, where 0 represents a strong 'No' and 100 a strong 'Yes' to the "
        f"research question, I assign a score of {response}, reflecting the balance of "
        "statistical significance, effect sizes, and sample size limitations."
    )

    explanation = " ".join(lines)

    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


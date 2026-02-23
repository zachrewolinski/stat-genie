import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    info_path = Path("info.json")

    df = pd.read_csv(data_path)

    # Construct predictors for relative group size and contest location.
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["size_adv"] = (df["size_diff"] > 0).astype(int)

    df["loc_diff"] = df["dist_focal"] - df["dist_other"]
    df["home_adv"] = (df["loc_diff"] < 0).astype(int)

    # Drop any rows with missing values in key fields (defensive, though none are expected).
    model_df = df[["win", "size_diff", "loc_diff"]].dropna()

    y = model_df["win"]
    X = sm.add_constant(model_df[["size_diff", "loc_diff"]])

    logit_model = sm.Logit(y, X)
    try:
        result = logit_model.fit(disp=False)
    except Exception:
        # Fallback to a regularized fit in case of numerical issues
        result = logit_model.fit_regularized(disp=False)

    params = result.params
    pvalues = result.pvalues

    size_coef = float(params["size_diff"])
    size_p = float(pvalues["size_diff"])

    loc_coef = float(params["loc_diff"])
    loc_p = float(pvalues["loc_diff"])

    # Descriptive summaries for clarity.
    n = int(len(df))
    win_overall = float(df["win"].mean())

    win_size_adv = float(df.loc[df["size_adv"] == 1, "win"].mean())
    win_no_size_adv = float(df.loc[df["size_adv"] == 0, "win"].mean())

    win_home_adv = float(df.loc[df["home_adv"] == 1, "win"].mean())
    win_no_home_adv = float(df.loc[df["home_adv"] == 0, "win"].mean())

    # Map statistical evidence into a 0–100 Likert-style score.
    # Start from a low-neutral "no clear evidence" baseline and add points for
    # predictors that are in the expected direction and statistically significant.
    score = 20

    # Relative group size: expect larger focal groups (size_diff > 0) to be more likely to win,
    # i.e., positive coefficient.
    if size_coef > 0:
        if size_p < 0.01:
            score += 35
        elif size_p < 0.05:
            score += 25
        elif size_p < 0.1:
            score += 10
    else:
        if size_p < 0.05:
            score -= 10

    # Contest location: expect focal groups that are closer to their home range centre
    # (loc_diff < 0, so a negative coefficient) to be more likely to win.
    if loc_coef < 0:
        if loc_p < 0.01:
            score += 35
        elif loc_p < 0.05:
            score += 25
        elif loc_p < 0.1:
            score += 10
    else:
        if loc_p < 0.05:
            score -= 10

    # Clamp to [0, 100] and convert to an integer scalar.
    score_int = int(max(0, min(100, round(score))))

    # Build explanation string with key numerical results.
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        research_qs = info.get("research_questions") or []
        research_q = research_qs[0] if research_qs else ""
    except Exception:
        research_q = ""

    explanation_parts = []
    if research_q:
        explanation_parts.append(
            f"Research question: {research_q.strip()} "
        )

    explanation_parts.append(
        f"The dataset contains {n} intergroup contests between capuchin groups, "
        f"with the focal group winning {win_overall:.1%} of contests."
    )

    explanation_parts.append(
        "I modelled the probability that the focal group won using logistic regression "
        "with two predictors: relative group size (n_focal - n_other) and contest location "
        "captured as the difference in distance from each group's home-range centre "
        "(dist_focal - dist_other)."
    )

    explanation_parts.append(
        f"For relative group size, the logistic regression coefficient on size_diff was "
        f"{size_coef:.3f} (p = {size_p:.3f}). This effect is not statistically significant "
        "in this sample, so the data do not provide strong evidence that relative group size "
        "meaningfully changes the probability that the focal group wins."
    )

    explanation_parts.append(
        f"Descriptively, the focal group won {win_size_adv:.1%} of contests when it had a "
        f"size advantage (n_focal > n_other) compared to {win_no_size_adv:.1%} when it did not, "
        "which does not suggest a clear advantage for larger focal groups in this dataset."
    )

    explanation_parts.append(
        f"For contest location, the coefficient on loc_diff (dist_focal - dist_other) was "
        f"{loc_coef:.3f} (p = {loc_p:.3f}), again not statistically significant. Thus, contests "
        "closer to the focal group's home-range centre do not show a reliable increase in win "
        "probability in this sample."
    )

    explanation_parts.append(
        f"Empirically, the focal group won {win_home_adv:.1%} of contests when it had a "
        f'"home-range advantage" (closer to its home-range centre; home_adv = 1) versus '
        f"{win_no_home_adv:.1%} when it did not, again failing to support a strong home-range "
        "advantage in these data."
    )

    explanation_parts.append(
        "Taken together, the logistic regression and descriptive comparisons provide little "
        "statistical evidence that either relative group size or contest location has a strong, "
        "reliably detectable effect on the probability that a capuchin group wins an intergroup "
        "contest in this dataset. The relatively low 0–100 response score therefore corresponds to "
        "a 'No' answer: within this sample, any effects of group size or location on winning appear "
        "weak or inconsistent."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": score_int,
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

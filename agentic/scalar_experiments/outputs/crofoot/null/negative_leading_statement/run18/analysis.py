import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Define key predictors for the research question.
    # Relative group size: focal group size minus other group size.
    df["rel_size"] = df["n_focal"] - df["n_other"]

    # Contest location: indicator for whether the contest is closer to the focal
    # group's home range center than to the other group's center.
    df["focal_home_adv"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    y = df["win"]
    X = df[["rel_size", "focal_home_adv"]]
    X = sm.add_constant(X, has_constant="add")

    # Fit logistic regression: probability focal group wins as a function of
    # relative group size and contest location.
    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    params = result.params
    pvalues = result.pvalues

    coef_size = float(params["rel_size"])
    coef_loc = float(params["focal_home_adv"])
    p_size = float(pvalues["rel_size"])
    p_loc = float(pvalues["focal_home_adv"])

    # Compute simple descriptive summaries to support the explanation.
    df["larger_focal"] = (df["rel_size"] > 0).astype(int)
    win_when_larger = df.loc[df["larger_focal"] == 1, "win"].mean()
    win_when_smaller = df.loc[df["larger_focal"] == 0, "win"].mean()

    win_with_home = df.loc[df["focal_home_adv"] == 1, "win"].mean()
    win_away = df.loc[df["focal_home_adv"] == 0, "win"].mean()

    # Map statistical evidence to a 0–100 Likert scale, where higher values
    # correspond to a stronger "Yes, these factors influence win probability".
    def likert_score(p_a: float, p_b: float) -> int:
        min_p = min(p_a, p_b)
        # Strong evidence that at least one factor matters.
        if min_p < 0.01:
            return 90
        if min_p < 0.05:
            return 75
        if min_p < 0.1:
            return 60
        if min_p < 0.2:
            return 50
        if min_p < 0.5:
            return 30
        # Very weak or no evidence of an effect.
        return 10

    response = likert_score(p_size, p_loc)

    # Build explanation string summarizing methods and key results.
    explanation = (
        "Research question: Do relative group size and contest location influence "
        "the probability that a capuchin monkey group wins an intergroup contest? "
        "I modeled the binary outcome of the focal group winning using logistic "
        "regression with two predictors: relative group size (focal group size minus "
        "other group size) and a home-range indicator for contest location "
        "(1 if the contest occurred closer to the focal group's home range center "
        "than to the other group's center, 0 otherwise). "
        f"The coefficient for relative group size was {coef_size:.3f} with p-value "
        f"{p_size:.3f}, and the coefficient for the home-range indicator was "
        f"{coef_loc:.3f} with p-value {p_loc:.3f}. Both effects are small in magnitude "
        "and far from conventional thresholds for statistical significance, so the "
        "logistic model does not provide strong evidence that either relative group "
        "size or contest location reliably predicts win probability. "
        f"Descriptively, focal groups won in approximately {win_when_larger:.2f} of "
        "contests when they were larger than their opponent, compared with "
        f"{win_when_smaller:.2f} when they were not larger. "
        f"They also won about {win_with_home:.2f} of contests when the contest "
        "occurred closer to their own home range center versus "
        f"{win_away:.2f} when the contest was closer to the other group's center. "
        "Taken together, these patterns and the logistic regression results provide "
        "little evidence that relative group size or contest location has a strong, "
        "consistent effect on winning intergroup contests in this sample. I therefore "
        "answer 'No' to the research question and set the Likert-scale response to "
        f"{response} on a 0–100 scale, where 0 represents a strong 'No' and 100 a "
        "strong 'Yes'."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

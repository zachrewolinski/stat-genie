import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    if not data_path.exists():
        raise FileNotFoundError("crofoot.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Outcome: 1 if focal group won, 0 otherwise.
    y = df["feature4"].astype(int)

    # Relative group size: focal minus other.
    size_diff = df["feature7"] - df["feature8"]

    # Contest location advantage: positive when focal group is closer
    # to its own home range center than the other group is to its own.
    loc_adv = df["feature6"] - df["feature5"]

    # Standardize predictors to ease interpretation and numerical stability.
    X = pd.DataFrame(
        {
            "size_diff_z": (size_diff - size_diff.mean()) / size_diff.std(ddof=0),
            "loc_adv_z": (loc_adv - loc_adv.mean()) / loc_adv.std(ddof=0),
        }
    )
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    # Extract key statistics.
    params = result.params
    pvalues = result.pvalues

    size_effect = params["size_diff_z"]
    size_p = float(pvalues["size_diff_z"])

    loc_effect = params["loc_adv_z"]
    loc_p = float(pvalues["loc_adv_z"])

    # Compute simple descriptive checks as additional evidence.
    df_extended = df.copy()
    df_extended["size_diff"] = size_diff
    df_extended["loc_adv"] = loc_adv

    # Probability of win when focal is larger vs smaller/equal.
    larger = df_extended["size_diff"] > 0
    win_rate_larger = df_extended.loc[larger, "feature4"].mean()
    win_rate_not_larger = df_extended.loc[~larger, "feature4"].mean()

    # Probability of win when focal has home-location advantage vs not.
    home_adv = df_extended["loc_adv"] > 0
    win_rate_home = df_extended.loc[home_adv, "feature4"].mean()
    win_rate_away = df_extended.loc[~home_adv, "feature4"].mean()

    # Map evidence to a Likert-style score (0-100).
    # Start from an agnostic midpoint and adjust based on significance and effect size.
    score = 50

    # Adjust for size effect.
    if size_p < 0.05:
        if size_effect > 0:
            score += 15
        else:
            score -= 5
    elif size_p < 0.10:
        if size_effect > 0:
            score += 5
        else:
            score -= 2

    # Adjust for location effect.
    if loc_p < 0.05:
        if loc_effect > 0:
            score += 20
        else:
            score -= 5
    elif loc_p < 0.10:
        if loc_effect > 0:
            score += 8
        else:
            score -= 3

    # Clamp score to [0, 100] and cast to int.
    score = int(max(0, min(100, round(score))))

    # Build explanation string summarizing the evidence.
    explanation_parts = []
    explanation_parts.append(
        "I modeled the probability that the focal capuchin group won an intergroup contest "
        "using logistic regression with the binary outcome (win vs. loss) regressed on "
        "two standardized predictors: relative group size (focal group size minus other "
        "group size) and a contest location advantage index (other group distance from its "
        "home-range center minus focal group distance from its own center)."
    )

    explanation_parts.append(
        f"In this model, the coefficient for standardized relative group size was "
        f"{size_effect:.2f} with p-value {size_p:.3f}, while the coefficient for the "
        f"standardized location advantage was {loc_effect:.2f} with p-value {loc_p:.3f}."
    )

    explanation_parts.append(
        "I also examined descriptive win rates: the focal group won "
        f"{win_rate_larger:.2%} of contests when it was larger than the opposing group, "
        f"compared to {win_rate_not_larger:.2%} when it was not larger; similarly, the focal "
        f"group won {win_rate_home:.2%} of contests when it was closer to its own home-range "
        f"center than the other group was to theirs, versus {win_rate_away:.2%} when the "
        "other group had the location advantage."
    )

    if score >= 60:
        overall = (
            "Taken together, these patterns provide evidence that relative group size "
            "and contest location do influence the probability that a capuchin group "
            "wins an intergroup contest, although the strength of the effects is "
            "moderate given the small sample size (58 contests)."
        )
    else:
        overall = (
            "Taken together, the limited sample and estimated effects do not provide "
            "strong, consistent evidence that relative group size and contest location "
            "meaningfully influence the probability of winning, so any such relationships "
            "should be interpreted cautiously."
        )

    explanation_parts.append(overall)

    explanation = " ".join(explanation_parts)

    conclusion = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


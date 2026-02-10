import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    df["win"] = df["feature4"].astype(int)

    # Relative group size (focal - other), using total number of individuals
    df["rel_size"] = df["feature7"] - df["feature8"]

    # Contest location advantage for focal:
    # positive when contest is closer to focal home-range center than to other's
    df["loc_advantage"] = df["feature6"] - df["feature5"]

    # Basic descriptive checks
    larger_mask = df["rel_size"] > 0
    smaller_mask = df["rel_size"] < 0
    equal_mask = df["rel_size"] == 0

    win_rate_larger = df.loc[larger_mask, "win"].mean()
    win_rate_smaller = df.loc[smaller_mask, "win"].mean()
    win_rate_equal = df.loc[equal_mask, "win"].mean()

    # Location categories: focal closer, other closer, roughly equal
    loc_diff = df["loc_advantage"]
    thresh = loc_diff.abs().median()
    focal_home = loc_diff > thresh
    other_home = loc_diff < -thresh
    neutral = ~focal_home & ~other_home

    win_rate_focal_home = df.loc[focal_home, "win"].mean()
    win_rate_other_home = df.loc[other_home, "win"].mean()
    win_rate_neutral = df.loc[neutral, "win"].mean()

    # Logistic regression: win ~ rel_size + loc_advantage (scaled per 100m)
    df["loc_advantage_100"] = df["loc_advantage"] / 100.0
    X = df[["rel_size", "loc_advantage_100"]]
    X = sm.add_constant(X)
    y = df["win"]

    logit_model = sm.Logit(y, X).fit(disp=False)
    params = logit_model.params
    pvalues = logit_model.pvalues

    # Heuristic evidence scoring on [-100, 100]
    # Start at neutral and adjust based on effect directions, magnitudes, and p-values.
    score = 0

    # Relative size effect: expect larger groups to be more likely to win.
    size_coef = params["rel_size"]
    size_p = pvalues["rel_size"]
    if size_coef > 0:
        if size_p < 0.01:
            score += 25
        elif size_p < 0.05:
            score += 18
        elif size_p < 0.1:
            score += 10
        else:
            score += 5
    elif size_coef < 0:
        if size_p < 0.05:
            score -= 20

    # Location advantage effect: expect positive coefficient (home advantage).
    loc_coef = params["loc_advantage_100"]
    loc_p = pvalues["loc_advantage_100"]
    if loc_coef > 0:
        if loc_p < 0.01:
            score += 40
        elif loc_p < 0.05:
            score += 30
        elif loc_p < 0.1:
            score += 18
        else:
            score += 8
    elif loc_coef < 0:
        if loc_p < 0.05:
            score -= 30

    # Descriptive reinforcement: if patterns are in expected directions, bump score.
    if win_rate_larger > win_rate_smaller:
        score += 5
    if win_rate_focal_home > win_rate_other_home:
        score += 10

    # Clip to Likert scale bounds
    score = int(max(-100, min(100, round(score))))

    # Write scalar conclusion for downstream consumption
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(score))


if __name__ == "__main__":
    main()


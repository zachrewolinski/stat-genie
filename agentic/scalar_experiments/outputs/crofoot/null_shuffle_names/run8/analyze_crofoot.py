import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    df["outcome"] = df["m_focal"]

    # Group sizes: descriptions in info.json indicate these mappings
    # f_other: number of individuals in focal group
    # win:    number of individuals in other group
    df["size_focal"] = df["f_other"]
    df["size_other"] = df["win"]
    df["size_diff"] = df["size_focal"] - df["size_other"]
    df["focal_larger"] = df["size_diff"] > 0

    # Contest location: distances from each group's home-range center
    # m_other: distance of focal group from center of its home range
    # n_focal: distance of other group from center of its home range
    df["dist_focal"] = df["m_other"]
    df["dist_other"] = df["n_focal"]
    df["dist_diff"] = df["dist_other"] - df["dist_focal"]  # >0 means focal is closer
    df["focal_closer"] = df["dist_diff"] > 0

    overall_win = df["outcome"].mean()

    def safe_mean(mask: pd.Series) -> float:
        sub = df.loc[mask, "outcome"]
        return float(sub.mean()) if len(sub) > 0 else float("nan")

    win_focal_larger = safe_mean(df["focal_larger"])
    win_focal_not_larger = safe_mean(~df["focal_larger"])

    win_focal_closer = safe_mean(df["focal_closer"])
    win_focal_not_closer = safe_mean(~df["focal_closer"])

    # Logistic regression with continuous predictors (standardized)
    X = df[["size_diff", "dist_diff"]].copy()
    X = (X - X.mean()) / X.std(ddof=0)
    X = sm.add_constant(X)
    y = df["outcome"]

    logit_model = sm.Logit(y, X).fit(disp=False)
    summary = logit_model.summary2().as_text()

    print("Overall focal win rate:", round(overall_win, 3))
    print()
    print("Win prob when focal group larger:", round(win_focal_larger, 3))
    print("Win prob when focal group <= other:", round(win_focal_not_larger, 3))
    print()
    print("Win prob when focal closer to its range center:", round(win_focal_closer, 3))
    print("Win prob when focal not closer:", round(win_focal_not_closer, 3))
    print()
    print("Logistic regression on outcome ~ size_diff + dist_diff (standardized predictors):")
    print(summary)


if __name__ == "__main__":
    main()


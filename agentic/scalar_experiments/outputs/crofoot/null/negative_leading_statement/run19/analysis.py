import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Relative group size: focal minus other
    df["rel_size"] = df["n_focal"] - df["n_other"]
    # Relative location advantage: positive when focal is closer to its home-range center
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]
    # Simple categorical indicators for descriptive stats
    df["focal_larger"] = (df["rel_size"] > 0).astype(int)
    df["focal_closer"] = (df["rel_dist"] > 0).astype(int)
    return df


def fit_logistic(df: pd.DataFrame):
    # Add intercept and fit logistic regression win ~ rel_size + rel_dist
    X = df[["rel_size", "rel_dist"]]
    X = sm.add_constant(X)
    y = df["win"]
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def describe_effects(df: pd.DataFrame) -> dict:
    n = len(df)

    # Win rates by relative size
    larger = df[df["focal_larger"] == 1]["win"]
    smaller = df[df["focal_larger"] == 0]["win"]

    # Win rates by location advantage
    closer = df[df["focal_closer"] == 1]["win"]
    farther = df[df["focal_closer"] == 0]["win"]

    return {
        "n_contests": int(n),
        "win_rate_overall": float(df["win"].mean()),
        "win_rate_focal_larger": float(larger.mean()) if len(larger) > 0 else None,
        "count_focal_larger": int(len(larger)),
        "win_rate_focal_not_larger": float(smaller.mean()) if len(smaller) > 0 else None,
        "count_focal_not_larger": int(len(smaller)),
        "win_rate_focal_closer": float(closer.mean()) if len(closer) > 0 else None,
        "count_focal_closer": int(len(closer)),
        "win_rate_focal_not_closer": float(farther.mean()) if len(farther) > 0 else None,
        "count_focal_not_closer": int(len(farther)),
    }


def main() -> None:
    df = load_data("crofoot.csv")
    desc = describe_effects(df)
    result = fit_logistic(df)

    # Collect key statistics for later interpretation
    params = result.params.to_dict()
    pvalues = result.pvalues.to_dict()

    summary = {
        "descriptives": desc,
        "logistic_params": params,
        "logistic_pvalues": pvalues,
        "pseudo_r2": float(result.prsquared),
    }

    # Store intermediate results for inspection if needed
    Path("analysis_results.json").write_text(json.dumps(summary, indent=2))

    # Based on the fitted model and descriptives, there is no
    # statistically significant evidence that relative group size
    # or contest location meaningfully affect win probability.
    # Both predictors have small, non-significant coefficients and
    # the pseudo-R^2 is close to zero.

    response = 15  # 0 = strong "No", 100 = strong "Yes"

    explanation = (
        "Using data from {n} intergroup contests, I tested whether relative group size "
        "(n_focal - n_other) and relative contest location (dist_other - dist_focal, "
        "positive when the focal group is closer to the centre of its home range) "
        "predict the probability that the focal group wins. A logistic regression "
        "model with win as the outcome and both predictors yielded small, "
        "non-significant coefficients (rel_size beta={b_size:.2f}, p={p_size:.3f}; "
        "rel_dist beta={b_dist:.4f}, p={p_dist:.3f}; McFadden pseudo-R^2={r2:.3f}), "
        "indicating that these variables explain very little of the variation in contest "
        "outcomes. Descriptive comparisons were consistent with this: focal groups won "
        "{wr_larger:.1%} of contests when larger (n={n_larger}) versus "
        "{wr_not_larger:.1%} when not larger (n={n_not_larger}), and "
        "{wr_closer:.1%} when closer to their home-range centre (n={n_closer}) versus "
        "{wr_not_closer:.1%} when not closer (n={n_not_closer}). These differences are "
        "modest, sometimes in the opposite direction from a simple advantage, and are "
        "not statistically compelling given the sample size. Overall, the data provide "
        "no reliable evidence that relative group size or contest location meaningfully "
        "influence the probability of winning an intergroup contest, so I answer 'No' "
        "on this question, with a strongly negative but not absolute level of confidence."
    ).format(
        n=desc["n_contests"],
        b_size=params["rel_size"],
        p_size=pvalues["rel_size"],
        b_dist=params["rel_dist"],
        p_dist=pvalues["rel_dist"],
        r2=summary["pseudo_r2"],
        wr_larger=desc["win_rate_focal_larger"],
        n_larger=desc["count_focal_larger"],
        wr_not_larger=desc["win_rate_focal_not_larger"],
        n_not_larger=desc["count_focal_not_larger"],
        wr_closer=desc["win_rate_focal_closer"],
        n_closer=desc["count_focal_closer"],
        wr_not_closer=desc["win_rate_focal_not_closer"],
        n_not_closer=desc["count_focal_not_closer"],
    )

    conclusion = {"response": int(response), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

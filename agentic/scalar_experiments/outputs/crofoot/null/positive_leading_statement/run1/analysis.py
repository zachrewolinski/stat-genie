import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Relative group size: focal minus other, and ratio
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["size_ratio"] = df["n_focal"] / df["n_other"]

    # Contest location: relative proximity to home range center
    df["dist_diff"] = df["dist_other"] - df["dist_focal"]

    # Center and scale predictors for stability
    for col in ["size_diff", "size_ratio", "dist_diff"]:
        df[f"c_{col}"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    y = df["win"]
    X = df[["c_size_diff", "c_size_ratio", "c_dist_diff"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X).fit(disp=False)

    summary = {
        "params": logit_model.params.to_dict(),
        "pvalues": logit_model.pvalues.to_dict(),
        "conf_int": logit_model.conf_int().to_dict(orient="index"),
        "n_obs": int(logit_model.nobs),
        "pseudo_r2": float(logit_model.prsquared),
    }

    # Evaluate strength of evidence:
    # Look at sign, p-value, and effect magnitudes
    pvals = logit_model.pvalues
    params = logit_model.params

    evidence_score = 0.0

    # Relative size: we expect larger focal group (size_diff/ratio) to increase win prob
    for var in ["c_size_diff", "c_size_ratio"]:
        if pvals[var] < 0.05 and params[var] > 0:
            evidence_score += 40.0  # strong positive evidence
        elif pvals[var] < 0.1 and params[var] > 0:
            evidence_score += 25.0  # moderate evidence
        elif pvals[var] < 0.2 and params[var] > 0:
            evidence_score += 10.0  # weak suggestion
        elif pvals[var] < 0.05 and params[var] < 0:
            evidence_score -= 30.0  # significant but opposite direction

    # Location: dist_diff > 0 means other group farther from its center
    # We expect focal more likely to win when other is farther from its home range center.
    if pvals["c_dist_diff"] < 0.05 and params["c_dist_diff"] > 0:
        evidence_score += 40.0
    elif pvals["c_dist_diff"] < 0.1 and params["c_dist_diff"] > 0:
        evidence_score += 25.0
    elif pvals["c_dist_diff"] < 0.2 and params["c_dist_diff"] > 0:
        evidence_score += 10.0
    elif pvals["c_dist_diff"] < 0.05 and params["c_dist_diff"] < 0:
        evidence_score -= 30.0

    # Incorporate overall pseudo R^2 as a modest adjustment
    if summary["pseudo_r2"] >= 0.2:
        evidence_score += 10.0
    elif summary["pseudo_r2"] >= 0.1:
        evidence_score += 5.0

    # Map evidence_score roughly into [0, 100]
    # Start from neutral 50 and shift; clamp to [0, 100]
    likert = int(round(np.clip(50.0 + evidence_score, 0.0, 100.0)))

    # Build textual explanation using key statistics
    explanation = {
        "n_obs": summary["n_obs"],
        "pseudo_r2": summary["pseudo_r2"],
        "params": summary["params"],
        "pvalues": summary["pvalues"],
        "conf_int": summary["conf_int"],
        "interpretation": (
            "Logistic regression models the probability that the focal capuchin group "
            "wins an intergroup contest as a function of relative group size "
            "(difference and ratio in group size) and contest location "
            "(difference in distance from each group to its home range center). "
            "The Likert score summarizes the strength and direction of statistical "
            "evidence that these factors influence win probability."
        ),
    }

    output = {
        "response": likert,
        "explanation_details": explanation,
    }

    Path("analysis_results.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()


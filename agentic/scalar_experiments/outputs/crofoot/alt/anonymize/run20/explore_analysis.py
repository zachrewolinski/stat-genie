import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Outcome: 1 if focal group won the contest.
    df["win"] = df["feature4"]

    # Relative group size: focal minus other group size.
    df["size_diff"] = df["feature7"] - df["feature8"]

    # Contest location: distance of each group from its home range center.
    # Positive dist_diff means focal group is farther from its center than the other group.
    df["dist_diff"] = df["feature5"] - df["feature6"]

    # Simple indicator: 1 if focal is closer to its own home range center than the other group is to theirs.
    df["focal_closer_home"] = (df["feature5"] < df["feature6"]).astype(int)

    y = df["win"]

    # Helper to fit a logistic regression and return summary info.
    def fit_logit(predictor_cols):
        X = df[predictor_cols].copy()
        X = sm.add_constant(X, has_constant="add")
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
        return result

    results = {}

    # Model 1: relative group size only.
    res_size = fit_logit(["size_diff"])
    results["size_only"] = {
        "params": res_size.params.to_dict(),
        "pvalues": res_size.pvalues.to_dict(),
        "llf": float(res_size.llf),
    }

    # Model 2: location (distance difference) only.
    res_loc = fit_logit(["dist_diff"])
    results["location_only"] = {
        "params": res_loc.params.to_dict(),
        "pvalues": res_loc.pvalues.to_dict(),
        "llf": float(res_loc.llf),
    }

    # Model 3: focal closer to home indicator only.
    res_loc_bin = fit_logit(["focal_closer_home"])
    results["location_binary_only"] = {
        "params": res_loc_bin.params.to_dict(),
        "pvalues": res_loc_bin.pvalues.to_dict(),
        "llf": float(res_loc_bin.llf),
    }

    # Model 4: both predictors plus their interaction.
    df["interaction"] = df["size_diff"] * df["dist_diff"]
    res_both = fit_logit(["size_diff", "dist_diff", "interaction"])
    results["size_and_location_with_interaction"] = {
        "params": res_both.params.to_dict(),
        "pvalues": res_both.pvalues.to_dict(),
        "llf": float(res_both.llf),
    }

    # Some descriptive contrasts to help interpretation.
    df["focal_larger"] = (df["size_diff"] > 0).astype(int)
    df["focal_much_larger"] = (df["size_diff"] >= 3).astype(int)
    df["focal_much_smaller"] = (df["size_diff"] <= -3).astype(int)

    desc = {}
    for col in ["focal_larger", "focal_much_larger", "focal_much_smaller", "focal_closer_home"]:
        grouped = df.groupby(col)["win"].agg(["mean", "count"]).reset_index()
        desc[col] = grouped.to_dict(orient="list")

    results["descriptives"] = desc

    # Print a JSON summary to stdout so we can inspect the results interactively.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


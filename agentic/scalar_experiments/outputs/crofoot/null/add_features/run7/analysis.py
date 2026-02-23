import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Keep rows with all relevant variables present
    df = df.dropna(
        subset=["win", "n_focal", "n_other", "dist_focal", "dist_other", "dyad"]
    )

    # Construct key predictors:
    #   - relative group size (focal - other): positive when focal group is larger
    #   - relative location (other_dist - focal_dist): positive when focal is closer to home
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["rel_location"] = df["dist_other"] - df["dist_focal"]

    # Simple binary indicators for descriptive win rates
    df["focal_larger"] = (df["n_focal"] > df["n_other"]).astype(int)
    df["focal_home_adv"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    # Standardize predictors for more interpretable coefficients
    X = df[["rel_group_size", "rel_location"]].astype(float).copy()
    for col in X.columns:
        std = X[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variation, center only to avoid division by zero
            X[col] = X[col] - X[col].mean()
        else:
            X[col] = (X[col] - X[col].mean()) / std

    X = sm.add_constant(X)
    y = df["win"].astype(float)

    # Fit logistic regression with cluster-robust SEs by dyad
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["dyad"]})

    params = {k: float(v) for k, v in result.params.to_dict().items()}
    pvalues = {k: float(v) for k, v in result.pvalues.to_dict().items()}

    # Descriptive win rates by size and location advantages
    size_stats = (
        df.groupby("focal_larger")["win"]
        .agg(["mean", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    loc_stats = (
        df.groupby("focal_home_adv")["win"]
        .agg(["mean", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    joint_stats = (
        df.groupby(["focal_larger", "focal_home_adv"])["win"]
        .agg(["mean", "count"])
        .reset_index()
        .to_dict(orient="records")
    )

    output = {
        "n_obs": int(result.nobs),
        "params": params,
        "pvalues": pvalues,
        "cov_type": result.cov_type,
        "win_rates_by_size": size_stats,
        "win_rates_by_location": loc_stats,
        "win_rates_by_size_and_location": joint_stats,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Construct key predictors
    df["size_diff"] = df["n_focal"] - df["n_other"]
    # Ratio-based measure of relative size; add a small epsilon to avoid
    # division by zero (not expected here, but kept for robustness).
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    df["log_size_ratio"] = (df["size_ratio"]).map(float).pipe(np.log)
    # Positive values mean the focal group is relatively closer to the center
    # of its home range than the opposing group is to its own center.
    df["loc_diff"] = df["dist_other"] - df["dist_focal"]

    # Standardize predictors for easier interpretation of coefficients
    for col in ["size_diff", "loc_diff", "log_size_ratio"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        df[col + "_z"] = (df[col] - mean) / std

    y = df["win"]

    def run_logit(predictor_cols, label):
        X = df[list(predictor_cols)]
        X = sm.add_constant(X)
        model = sm.Logit(y, X)
        res = model.fit(disp=False)
        print(f"\n=== Model: {label} ===")
        print(res.summary())
        print("\nCoefficients:")
        for name in res.params.index:
            print(f"  {name}: coef={res.params[name]:.3f}, p={res.pvalues[name]:.4f}")
        return res

    # Model 1: size difference + location difference
    res_diff = run_logit(["size_diff_z", "loc_diff_z"], "win ~ size_diff_z + loc_diff_z")

    # Model 2: log size ratio + location difference
    res_ratio = run_logit(
        ["log_size_ratio_z", "loc_diff_z"],
        "win ~ log_size_ratio_z + loc_diff_z",
    )

    # Model 3: main effects + interaction
    df["interaction_z"] = df["size_diff_z"] * df["loc_diff_z"]
    res_interaction = run_logit(
        ["size_diff_z", "loc_diff_z", "interaction_z"],
        "win ~ size_diff_z * loc_diff_z",
    )

    print("\nEffect direction summary (Model 1):")
    size_sign = "positive" if res_diff.params["size_diff_z"] > 0 else "negative"
    loc_sign = "positive" if res_diff.params["loc_diff_z"] > 0 else "negative"
    print(f"  size_diff_z: {size_sign}")
    print(f"  loc_diff_z: {loc_sign}")

    # Descriptive summaries
    print("\n=== Descriptive summaries ===")
    print("Mean predictors by outcome (win):")
    print(
        df.groupby("win")[["size_diff", "size_ratio", "loc_diff"]].mean().rename(
            columns={
                "size_diff": "mean_size_diff",
                "size_ratio": "mean_size_ratio",
                "loc_diff": "mean_loc_diff",
            }
        )
    )
    print("\nCorrelations with winning:")
    corr = df[["win", "size_diff", "size_ratio", "loc_diff"]].corr()
    print(corr["win"])


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Construct key predictors
    df["rel_size"] = df["n_focal"] - df["n_other"]  # positive if focal larger
    df["rel_loc"] = df["dist_other"] - df["dist_focal"]  # positive if focal closer to its center than other is to its

    # Center predictors to aid interpretation
    for col in ["rel_size", "rel_loc"]:
        df[col + "_c"] = df[col] - df[col].mean()

    y = df["win"]

    # Model 1: relative group size only
    X1 = sm.add_constant(df[["rel_size_c"]])
    model1 = sm.Logit(y, X1).fit(disp=False)

    # Model 2: relative location only
    X2 = sm.add_constant(df[["rel_loc_c"]])
    model2 = sm.Logit(y, X2).fit(disp=False)

    # Model 3: both predictors
    X3 = sm.add_constant(df[["rel_size_c", "rel_loc_c"]])
    model3 = sm.Logit(y, X3).fit(disp=False)

    summary = {
        "n_obs": int(len(df)),
        "model1": {
            "params": model1.params.to_dict(),
            "pvalues": model1.pvalues.to_dict(),
            "llf": float(model1.llf),
        },
        "model2": {
            "params": model2.params.to_dict(),
            "pvalues": model2.pvalues.to_dict(),
            "llf": float(model2.llf),
        },
        "model3": {
            "params": model3.params.to_dict(),
            "pvalues": model3.pvalues.to_dict(),
            "llf": float(model3.llf),
        },
    }

    Path("analysis_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Construct predictors: relative group size and relative location advantage.
    # Relative group size: focal size minus other size.
    df["rel_size"] = df["n_focal"] - df["n_other"]

    # Location advantage: focal closer to its range center than other is to its own.
    df["loc_adv"] = df["dist_focal"] - df["dist_other"]

    # Standardize predictors for interpretability in the regression.
    for col in ["rel_size", "loc_adv"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[f"z_{col}"] = df[col] - mean
        else:
            df[f"z_{col}"] = (df[col] - mean) / std

    # Logistic regression: probability focal group wins.
    y = df["win"]
    X = df[["z_rel_size", "z_loc_adv"]]
    X = sm.add_constant(X)
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    summary = {
        "params": result.params.to_dict(),
        "pvalues": result.pvalues.to_dict(),
        "llf": float(result.llf),
        "nobs": int(result.nobs),
    }

    # Save a lightweight JSON summary for inspection if needed.
    Path("model_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["feature4"]

    # Relative group size: focal size minus other group size.
    df["rel_group_size"] = df["feature7"] - df["feature8"]

    # Contest location: relative home-range advantage.
    # Positive values indicate the focal group is closer to its home range center.
    df["delta_distance"] = df["feature6"] - df["feature5"]
    df["focal_home_advantage"] = (df["feature5"] < df["feature6"]).astype(int)

    # Design matrices for several logistic regression specifications.
    models = {}

    # Model 1: Relative group size only.
    X1 = sm.add_constant(df[["rel_group_size"]])
    models["rel_size_only"] = sm.Logit(y, X1).fit(disp=False)

    # Model 2: Relative location (continuous) only.
    X2 = sm.add_constant(df[["delta_distance"]])
    models["location_only"] = sm.Logit(y, X2).fit(disp=False)

    # Model 3: Both predictors (continuous).
    X3 = sm.add_constant(df[["rel_group_size", "delta_distance"]])
    models["both_continuous"] = sm.Logit(y, X3).fit(disp=False)

    # Model 4: Relative size and a simple home-field indicator.
    X4 = sm.add_constant(df[["rel_group_size", "focal_home_advantage"]])
    models["size_plus_home_indicator"] = sm.Logit(y, X4).fit(disp=False)

    # Collect core statistics for manual inspection.
    summary = {}
    for name, model in models.items():
        params = model.params.to_dict()
        pvalues = model.pvalues.to_dict()
        odds_ratios = {k: float(np.exp(v)) for k, v in params.items()}
        summary[name] = {
            "n_obs": int(model.nobs),
            "llf": float(model.llf),
            "aic": float(model.aic),
            "params": params,
            "pvalues": pvalues,
            "odds_ratios": odds_ratios,
        }

    # Print a compact JSON summary to stdout for interpretation.
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


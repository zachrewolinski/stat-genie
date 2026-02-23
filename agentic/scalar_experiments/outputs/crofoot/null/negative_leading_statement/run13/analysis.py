import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Construct relative predictors
    df["rel_size"] = df["n_focal"] - df["n_other"]
    # Positive when focal is closer to its home range center
    df["home_adv"] = df["dist_other"] - df["dist_focal"]

    # Binary advantage indicators
    df["size_adv_focal"] = (df["rel_size"] > 0).astype(int)
    df["home_adv_focal"] = (df["home_adv"] > 0).astype(int)

    # Basic descriptive summaries
    n_rows = len(df)
    win_rate = df["win"].mean()

    # Categorical summaries for readability
    df["size_cat"] = np.select(
        [df["rel_size"] > 0, df["rel_size"] < 0],
        ["focal_larger", "focal_smaller"],
        default="same_size",
    )
    df["home_cat"] = np.select(
        [df["home_adv"] > 0, df["home_adv"] < 0],
        ["focal_closer", "focal_farther"],
        default="same_distance",
    )

    win_by_size = df.groupby("size_cat")["win"].agg(["mean", "count"])
    win_by_home = df.groupby("home_cat")["win"].agg(["mean", "count"])

    # Simple contingency tables
    size_ct = pd.crosstab(df["size_adv_focal"], df["win"])
    home_ct = pd.crosstab(df["home_adv_focal"], df["win"])

    # Logistic regression: win ~ rel_size + home_adv
    X = df[["rel_size", "home_adv"]]
    X = sm.add_constant(X)
    y = df["win"]
    logit_model = sm.Logit(y, X).fit(disp=False)

    # Also individual models
    X_size = sm.add_constant(df[["rel_size"]])
    model_size = sm.Logit(y, X_size).fit(disp=False)

    X_home = sm.add_constant(df[["home_adv"]])
    model_home = sm.Logit(y, X_home).fit(disp=False)

    # Binary-predictor models
    X_size_bin = sm.add_constant(df[["size_adv_focal"]])
    model_size_bin = sm.Logit(y, X_size_bin).fit(disp=False)

    X_home_bin = sm.add_constant(df[["home_adv_focal"]])
    model_home_bin = sm.Logit(y, X_home_bin).fit(disp=False)

    results = {
        "n_rows": int(n_rows),
        "overall_win_rate": float(win_rate),
        "win_by_size": win_by_size.reset_index().to_dict(orient="records"),
        "win_by_home": win_by_home.reset_index().to_dict(orient="records"),
        "size_adv_crosstab": size_ct.to_dict(),
        "home_adv_crosstab": home_ct.to_dict(),
        "logit_full": {
            "params": {k: float(v) for k, v in logit_model.params.items()},
            "pvalues": {k: float(v) for k, v in logit_model.pvalues.items()},
        },
        "logit_size_only": {
            "params": {k: float(v) for k, v in model_size.params.items()},
            "pvalues": {k: float(v) for k, v in model_size.pvalues.items()},
        },
        "logit_home_only": {
            "params": {k: float(v) for k, v in model_home.params.items()},
            "pvalues": {k: float(v) for k, v in model_home.pvalues.items()},
        },
        "logit_size_binary": {
            "params": {k: float(v) for k, v in model_size_bin.params.items()},
            "pvalues": {k: float(v) for k, v in model_size_bin.pvalues.items()},
        },
        "logit_home_binary": {
            "params": {k: float(v) for k, v in model_home_bin.params.items()},
            "pvalues": {k: float(v) for k, v in model_home_bin.pvalues.items()},
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

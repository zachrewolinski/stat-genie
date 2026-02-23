import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["m_focal"]

    # Group sizes
    focal_size = df["f_other"]
    other_size = df["win"]
    df["rel_size"] = focal_size - other_size  # positive when focal group is larger

    # Contest location: distance of each group from the center of its own home range
    df["focal_distance"] = df["m_other"]
    df["other_distance"] = df["n_focal"]
    df["distance_diff"] = df["other_distance"] - df["focal_distance"]
    df["focal_closer"] = (df["focal_distance"] < df["other_distance"]).astype(int)

    # Model 1: effect of relative group size
    X_size = sm.add_constant(df[["rel_size"]])
    model_size = sm.Logit(y, X_size).fit(disp=False)

    # Model 2: effect of contest location (continuous difference and indicator)
    X_loc = sm.add_constant(df[["distance_diff", "focal_closer"]])
    model_loc = sm.Logit(y, X_loc).fit(disp=False)

    # Model 3: combined model
    X_both = sm.add_constant(df[["rel_size", "distance_diff", "focal_closer"]])
    model_both = sm.Logit(y, X_both).fit(disp=False)

    def summarize_model(name: str, model) -> dict:
        params = model.params
        pvalues = model.pvalues
        conf_int = model.conf_int()

        summary = {
            "llf": float(model.llf),
            "aic": float(model.aic),
            "bic": float(model.bic),
            "nobs": int(model.nobs),
            "params": {},
        }
        for term in params.index:
            summary["params"][term] = {
                "coef": float(params[term]),
                "pvalue": float(pvalues[term]),
                "conf_int_95": [float(conf_int.loc[term, 0]), float(conf_int.loc[term, 1])],
                "odds_ratio": float(np.exp(params[term])),
            }
        return {name: summary}

    results = {}
    results.update(summarize_model("size_only", model_size))
    results.update(summarize_model("location_only", model_loc))
    results.update(summarize_model("combined", model_both))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

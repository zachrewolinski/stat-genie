import json

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Construct key predictors
    # Relative group size: positive when focal group is larger
    df["rel_size"] = df["n_focal"] - df["n_other"]

    # Contest location advantage: positive when contest is closer to focal group's home-range centre
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]

    # Standardise predictors for stability and comparability
    for col in ["rel_size", "loc_adv"]:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    y = df["win"]

    # Base model: win ~ rel_size_z + loc_adv_z
    X_main = sm.add_constant(df[["rel_size_z", "loc_adv_z"]])
    model_main = sm.Logit(y, X_main)
    result_main = model_main.fit(disp=False)

    # Interaction model: win ~ rel_size_z * loc_adv_z
    df["interaction_z"] = df["rel_size_z"] * df["loc_adv_z"]
    X_int = sm.add_constant(df[["rel_size_z", "loc_adv_z", "interaction_z"]])
    model_int = sm.Logit(y, X_int)
    result_int = model_int.fit(disp=False)

    # Single-predictor models for robustness checks
    X_size = sm.add_constant(df[["rel_size_z"]])
    result_size = sm.Logit(y, X_size).fit(disp=False)

    X_loc = sm.add_constant(df[["loc_adv_z"]])
    result_loc = sm.Logit(y, X_loc).fit(disp=False)

    # Basic descriptive comparison of predictors between wins and losses
    # (Optional descriptive summaries omitted from JSON output to keep it simple)

    # Collect key statistics for quick inspection
    summary = {
        "n_obs": int(result_main.nobs),
        "main_model": {
            "params": result_main.params.to_dict(),
            "pvalues": result_main.pvalues.to_dict(),
            "prsquared": float(result_main.prsquared),
        },
        "size_only_model": {
            "params": result_size.params.to_dict(),
            "pvalues": result_size.pvalues.to_dict(),
            "prsquared": float(result_size.prsquared),
        },
        "location_only_model": {
            "params": result_loc.params.to_dict(),
            "pvalues": result_loc.pvalues.to_dict(),
            "prsquared": float(result_loc.prsquared),
        },
        "interaction_model": {
            "params": result_int.params.to_dict(),
            "pvalues": result_int.pvalues.to_dict(),
            "prsquared": float(result_int.prsquared),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

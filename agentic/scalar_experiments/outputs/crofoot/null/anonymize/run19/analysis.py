import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("crofoot.csv")
    df = pd.read_csv(data_path)

    # Derived predictors
    df["rel_group_size"] = df["feature7"] - df["feature8"]
    df["focal_closer"] = (df["feature5"] < df["feature6"]).astype(int)
    df["rel_distance"] = df["feature5"] - df["feature6"]

    results = {}

    # Model 1: contest location as a binary "home advantage" indicator
    formula1 = "feature4 ~ rel_group_size + focal_closer"
    model1 = smf.glm(formula=formula1, data=df, family=sm.families.Binomial())
    res1 = model1.fit()

    results["model_home_indicator"] = {
        "n_obs": int(res1.nobs),
        "params": res1.params.to_dict(),
        "pvalues": res1.pvalues.to_dict(),
        "conf_int": {
            name: {"lower": float(ci[0]), "upper": float(ci[1])}
            for name, ci in zip(res1.params.index, res1.conf_int().values)
        },
        "aic": float(res1.aic),
        "bic": float(res1.bic),
        "deviance": float(res1.deviance),
        "null_deviance": float(res1.null_deviance),
    }

    # Model 2: contest location as continuous distance difference
    formula2 = "feature4 ~ rel_group_size + rel_distance"
    model2 = smf.glm(formula=formula2, data=df, family=sm.families.Binomial())
    res2 = model2.fit()

    results["model_distance_difference"] = {
        "n_obs": int(res2.nobs),
        "params": res2.params.to_dict(),
        "pvalues": res2.pvalues.to_dict(),
        "conf_int": {
            name: {"lower": float(ci[0]), "upper": float(ci[1])}
            for name, ci in zip(res2.params.index, res2.conf_int().values)
        },
        "aic": float(res2.aic),
        "bic": float(res2.bic),
        "deviance": float(res2.deviance),
        "null_deviance": float(res2.null_deviance),
    }

    with Path("analysis_results.json").open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("panda_nuts.csv")
    df = df.copy()
    df["rate"] = df["nuts_opened"] / df["seconds"]

    # Ensure categorical types
    df["sex"] = df["sex"].astype("category")
    df["help"] = df["help"].astype("category")

    # Poisson model with exposure (seconds) and cluster-robust SE by chimpanzee
    model = smf.glm(
        formula="nuts_opened ~ age + C(sex) + C(help)",
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["seconds"]),
    )
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

    # Extract coefficients and p-values
    params = res.params
    pvals = res.pvalues

    # Compute percent change in rate for 1-year age increase and categorical contrasts
    effects = {}
    for name in params.index:
        if name == "Intercept":
            continue
        effects[name] = {
            "coef": float(params[name]),
            "p_value": float(pvals[name]),
            "rate_multiplier": float(np.exp(params[name])),
            "percent_change": float((np.exp(params[name]) - 1) * 100),
        }

    # Descriptives
    desc = {
        "n_rows": int(len(df)),
        "mean_rate": float(df["rate"].mean()),
        "median_rate": float(df["rate"].median()),
        "by_sex": df.groupby("sex")["rate"].agg(["mean", "median", "count"]).to_dict(),
        "by_help": df.groupby("help")["rate"].agg(["mean", "median", "count"]).to_dict(),
        "age_range": [float(df["age"].min()), float(df["age"].max())],
    }

    output = {
        "effects": effects,
        "model_summary": {
            "aic": float(res.aic),
            "llf": float(res.llf),
            "df_model": int(res.df_model),
        },
        "descriptives": desc,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()

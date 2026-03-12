import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).parent
    data_path = base_path / "caschools.csv"
    info_path = base_path / "info.json"

    df = pd.read_csv(data_path)

    # Student–teacher ratio: students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    # Average test score as a simple composite of reading and math
    df["avgscore"] = df[["read", "math"]].mean(axis=1)

    # Drop any rows with missing key variables (should be none, but for safety)
    key_cols = ["stratio", "avgscore", "read", "math", "income", "english", "lunch", "calworks", "expenditure"]
    df = df.dropna(subset=[c for c in key_cols if c in df.columns])

    results = {}

    # Simple Pearson correlations
    for score_var in ["avgscore", "read", "math"]:
        corr = df["stratio"].corr(df[score_var])
        results[f"corr_stratio_{score_var}"] = corr

    # Bivariate OLS: score ~ stratio
    ols_simple = {}
    for score_var in ["avgscore", "read", "math"]:
        y = df[score_var]
        X = sm.add_constant(df["stratio"])
        model = sm.OLS(y, X).fit()
        coef = model.params["stratio"]
        pval = model.pvalues["stratio"]
        ols_simple[score_var] = {"coef": float(coef), "pval": float(pval), "r2": float(model.rsquared)}

    results["ols_simple"] = ols_simple

    # Multiple regression with key demographic and resource controls
    controls = ["income", "english", "lunch", "calworks", "expenditure"]
    multi = {}
    for score_var in ["avgscore", "read", "math"]:
        cols = ["stratio"] + [c for c in controls if c in df.columns]
        X = sm.add_constant(df[cols])
        y = df[score_var]
        model = sm.OLS(y, X).fit()
        coef = model.params["stratio"]
        pval = model.pvalues["stratio"]
        multi[score_var] = {
            "coef": float(coef),
            "pval": float(pval),
            "r2": float(model.rsquared),
        }

    results["ols_multi"] = multi

    # Heuristic summary to aid final Likert rating, but not the final output file.
    summary = {
        "corr": {k: round(v, 4) for k, v in results.items() if k.startswith("corr_")},
        "ols_simple": ols_simple,
        "ols_multi": multi,
    }

    with open(base_path / "analysis_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()


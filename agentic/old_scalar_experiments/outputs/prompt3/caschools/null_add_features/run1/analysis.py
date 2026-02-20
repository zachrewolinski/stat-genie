import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in key columns, if any
    df_model = df[["testscr", "stratio", "income", "english", "lunch"]].dropna()

    # Simple association: correlation and simple OLS
    corr = df_model["stratio"].corr(df_model["testscr"])

    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()

    # Multiple regression controlling for some key demographics
    X_controls = df_model[["stratio", "income", "english", "lunch"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["testscr"], X_controls).fit()

    results = {
        "n_obs": int(df_model.shape[0]),
        "corr_stratio_testscr": float(corr),
        "simple_coef_stratio": float(model_simple.params["stratio"]),
        "simple_pvalue_stratio": float(model_simple.pvalues["stratio"]),
        "simple_r2": float(model_simple.rsquared),
        "controls_coef_stratio": float(model_controls.params["stratio"]),
        "controls_pvalue_stratio": float(model_controls.pvalues["stratio"]),
        "controls_r2": float(model_controls.rsquared),
        "mean_stratio": float(df_model["stratio"].mean()),
        "std_stratio": float(df_model["stratio"].std()),
        "mean_testscr": float(df_model["testscr"].mean()),
        "std_testscr": float(df_model["testscr"].std()),
    }

    # Save numeric results for inspection
    out_path = Path("analysis_results.json")
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables for the research question.
    # Student–teacher ratio: students per teacher.
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores.
    df["score"] = df[["read", "math"]].mean(axis=1)

    # Keep only rows with complete data on variables used in analysis.
    core_cols = ["stratio", "score", "income", "english", "lunch", "calworks"]
    df_model = df[core_cols].dropna().copy()

    # Simple correlation between student–teacher ratio and test scores.
    r, p_value_corr = stats.pearsonr(df_model["stratio"], df_model["score"])

    # Simple OLS: score ~ stratio.
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["score"], X_simple).fit()
    coef_stratio_simple = float(model_simple.params["stratio"])
    p_value_stratio_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression controlling for key district characteristics.
    X_controls = df_model[["stratio", "income", "english", "lunch", "calworks"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["score"], X_controls).fit()
    coef_stratio_controls = float(model_controls.params["stratio"])
    p_value_stratio_controls = float(model_controls.pvalues["stratio"])
    r2_controls = float(model_controls.rsquared)

    # Bundle key summary statistics for inspection.
    summary = {
        "n_obs": int(df_model.shape[0]),
        "corr_stratio_score": float(r),
        "corr_p_value": float(p_value_corr),
        "simple_model": {
            "coef_stratio": coef_stratio_simple,
            "p_value_stratio": p_value_stratio_simple,
            "r_squared": r2_simple,
        },
        "controls_model": {
            "coef_stratio": coef_stratio_controls,
            "p_value_stratio": p_value_stratio_controls,
            "r_squared": r2_controls,
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError("caschools.csv not found in current directory")

    df = pd.read_csv(data_path)

    # Construct key variables for the analysis
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used below (should be rare/nonexistent)
    base_vars = ["stratio", "testscr", "income", "english", "lunch", "calworks"]
    df_model = df.dropna(subset=base_vars).copy()

    # Correlation between student-teacher ratio and test scores
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Simple bivariate regression
    model_simple = smf.ols("testscr ~ stratio", data=df_model).fit(cov_type="HC3")

    # Multiple regression with key socioeconomic controls
    model_controls = smf.ols(
        "testscr ~ stratio + income + english + lunch + calworks",
        data=df_model,
    ).fit(cov_type="HC3")

    # Robustness check: trim extreme values of the student-teacher ratio
    lower_q = df_model["stratio"].quantile(0.01)
    upper_q = df_model["stratio"].quantile(0.99)
    df_trim = df_model[(df_model["stratio"] >= lower_q) & (df_model["stratio"] <= upper_q)].copy()

    r_trim, p_corr_trim = stats.pearsonr(df_trim["stratio"], df_trim["testscr"])
    model_simple_trim = smf.ols("testscr ~ stratio", data=df_trim).fit(cov_type="HC3")

    results = {
        "n_obs": int(df_model.shape[0]),
        "stratio": {
            "mean": float(df_model["stratio"].mean()),
            "std": float(df_model["stratio"].std()),
        },
        "testscr": {
            "mean": float(df_model["testscr"].mean()),
            "std": float(df_model["testscr"].std()),
        },
        "correlation": {
            "r": float(r),
            "p_value": float(p_corr),
        },
        "trimmed_correlation": {
            "r": float(r_trim),
            "p_value": float(p_corr_trim),
            "lower_q_stratio": float(lower_q),
            "upper_q_stratio": float(upper_q),
            "n_obs_trimmed": int(df_trim.shape[0]),
        },
        "simple_regression": {
            "coef_stratio": float(model_simple.params["stratio"]),
            "se_stratio": float(model_simple.bse["stratio"]),
            "p_value_stratio": float(model_simple.pvalues["stratio"]),
            "r_squared": float(model_simple.rsquared),
        },
        "simple_regression_trimmed": {
            "coef_stratio": float(model_simple_trim.params["stratio"]),
            "se_stratio": float(model_simple_trim.bse["stratio"]),
            "p_value_stratio": float(model_simple_trim.pvalues["stratio"]),
            "r_squared": float(model_simple_trim.rsquared),
        },
        "controls_regression": {
            "coef_stratio": float(model_controls.params["stratio"]),
            "se_stratio": float(model_controls.bse["stratio"]),
            "p_value_stratio": float(model_controls.pvalues["stratio"]),
            "r_squared": float(model_controls.rsquared),
        },
    }

    # Save a compact JSON summary to inspect from the outside
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also print a brief human-readable summary
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

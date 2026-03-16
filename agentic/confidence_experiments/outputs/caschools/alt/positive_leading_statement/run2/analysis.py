import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Student-teacher ratio: higher values mean more students per teacher.
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0
    df["comp_per_student"] = df["computer"] / df["students"]

    # Basic correlations
    corr_read = df["stratio"].corr(df["read"])
    corr_math = df["stratio"].corr(df["math"])
    corr_avg = df["stratio"].corr(df["avg_score"])

    # Simple linear regressions
    mod_read_simple = smf.ols("read ~ stratio", data=df).fit()
    mod_math_simple = smf.ols("math ~ stratio", data=df).fit()
    mod_avg_simple = smf.ols("avg_score ~ stratio", data=df).fit()

    # Multiple regression controlling for key covariates
    formula_controls = (
        "avg_score ~ stratio + income + calworks + lunch + english "
        "+ expenditure + comp_per_student"
    )
    mod_avg_controls = smf.ols(formula_controls, data=df).fit()

    results = {
        "n_obs": int(df.shape[0]),
        "correlations": {
            "read_vs_stratio": corr_read,
            "math_vs_stratio": corr_math,
            "avg_vs_stratio": corr_avg,
        },
        "simple_regression": {
            "read": {
                "coef_stratio": mod_read_simple.params.get("stratio", np.nan),
                "pvalue_stratio": mod_read_simple.pvalues.get("stratio", np.nan),
                "r2": mod_read_simple.rsquared,
            },
            "math": {
                "coef_stratio": mod_math_simple.params.get("stratio", np.nan),
                "pvalue_stratio": mod_math_simple.pvalues.get("stratio", np.nan),
                "r2": mod_math_simple.rsquared,
            },
            "avg": {
                "coef_stratio": mod_avg_simple.params.get("stratio", np.nan),
                "pvalue_stratio": mod_avg_simple.pvalues.get("stratio", np.nan),
                "r2": mod_avg_simple.rsquared,
            },
        },
        "controls_regression": {
            "avg": {
                "coef_stratio": mod_avg_controls.params.get("stratio", np.nan),
                "pvalue_stratio": mod_avg_controls.pvalues.get("stratio", np.nan),
                "r2": mod_avg_controls.rsquared,
            }
        },
    }

    # Print a concise summary for human inspection.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_analysis() -> dict:
    """
    Load the caschools data, construct a student-teacher ratio,
    and run regressions of academic performance on this ratio.

    Returns a dictionary with the key summary statistics needed
    to write a substantive conclusion.
    """
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError("caschools.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Construct student-teacher ratio and an overall performance score.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    # Drop any rows with missing values in key columns, just in case.
    key_cols = ["stratio", "avg_score"]
    control_candidates = ["income", "english", "lunch", "calworks", "expenditure"]
    controls = [c for c in control_candidates if c in df.columns]
    key_cols.extend(controls)
    df_model = df.dropna(subset=key_cols)

    # Simple bivariate regression: avg_score ~ stratio
    y_avg = df_model["avg_score"]
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(y_avg, X_simple).fit()

    coef_simple = float(model_simple.params["stratio"])
    se_simple = float(model_simple.bse["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Also check reading and math scores separately.
    y_read = df_model["read"]
    model_read = sm.OLS(y_read, X_simple).fit()
    coef_read = float(model_read.params["stratio"])
    p_read = float(model_read.pvalues["stratio"])

    y_math = df_model["math"]
    model_math = sm.OLS(y_math, X_simple).fit()
    coef_math = float(model_math.params["stratio"])
    p_math = float(model_math.pvalues["stratio"])

    # Multiple regression with available controls, if any.
    results = {
        "n_obs": int(df_model.shape[0]),
        "coef_simple": coef_simple,
        "se_simple": se_simple,
        "p_simple": p_simple,
        "r2_simple": r2_simple,
        "coef_read": coef_read,
        "p_read": p_read,
        "coef_math": coef_math,
        "p_math": p_math,
        "controls_used": controls,
    }

    if controls:
        X_multi = sm.add_constant(df_model[["stratio"] + controls])
        model_multi = sm.OLS(y_avg, X_multi).fit()

        coef_multi = float(model_multi.params["stratio"])
        se_multi = float(model_multi.bse["stratio"])
        p_multi = float(model_multi.pvalues["stratio"])
        r2_multi = float(model_multi.rsquared)

        results.update(
            {
                "coef_multi": coef_multi,
                "se_multi": se_multi,
                "p_multi": p_multi,
                "r2_multi": r2_multi,
            }
        )

    return results


def main() -> None:
    res = run_analysis()

    print("Number of observations used:", res["n_obs"])
    print(
        "Simple regression avg_score ~ student-teacher ratio:",
        f"coef={res['coef_simple']:.3f},",
        f"se={res['se_simple']:.3f},",
        f"p={res['p_simple']:.3g},",
        f"R^2={res['r2_simple']:.3f}",
    )

    if "coef_multi" in res:
        print(
            "Multiple regression with controls",
            f"({', '.join(res['controls_used'])}):",
            f"coef={res['coef_multi']:.3f},",
            f"se={res['se_multi']:.3f},",
            f"p={res['p_multi']:.3g},",
            f"R^2={res['r2_multi']:.3f}",
        )

    print(
        "Reading score regression (read ~ ratio):",
        f"coef={res['coef_read']:.3f},",
        f"p={res['p_read']:.3g}",
    )
    print(
        "Math score regression (math ~ ratio):",
        f"coef={res['coef_math']:.3f},",
        f"p={res['p_math']:.3g}",
    )


if __name__ == "__main__":
    main()

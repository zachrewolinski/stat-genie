from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "caschools.csv"

    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    # Drop any rows with missing values in variables used below
    vars_basic = ["stratio", "avg_score"]
    vars_controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    df_basic = df[vars_basic].dropna()
    df_controls = df[vars_basic + vars_controls].dropna()

    print("Number of districts (basic):", len(df_basic))
    print("Number of districts (with controls):", len(df_controls))
    print()

    # Correlation between student-teacher ratio and achievement
    r, p = stats.pearsonr(df_basic["stratio"], df_basic["avg_score"])
    print("Pearson correlation between student-teacher ratio and average score:")
    print(f"  r = {r:.3f}, p-value = {p:.3g}")
    print()

    # Simple linear regression: avg_score ~ stratio
    X_simple = sm.add_constant(df_basic["stratio"])
    y_simple = df_basic["avg_score"]
    model_simple = sm.OLS(y_simple, X_simple).fit()
    coef_stratio_simple = model_simple.params["stratio"]
    se_stratio_simple = model_simple.bse["stratio"]
    p_stratio_simple = model_simple.pvalues["stratio"]

    print("OLS regression: avg_score ~ student-teacher ratio")
    print(f"  Coefficient on ratio = {coef_stratio_simple:.3f}")
    print(f"  Standard error      = {se_stratio_simple:.3f}")
    print(f"  p-value             = {p_stratio_simple:.3g}")
    print()

    # Multiple regression with key demographic and resource controls
    X_controls = df_controls[["stratio"] + vars_controls]
    X_controls = sm.add_constant(X_controls)
    y_controls = df_controls["avg_score"]
    model_controls = sm.OLS(y_controls, X_controls).fit()

    coef_stratio_ctrl = model_controls.params["stratio"]
    se_stratio_ctrl = model_controls.bse["stratio"]
    p_stratio_ctrl = model_controls.pvalues["stratio"]

    print(
        "OLS regression with controls: "
        "avg_score ~ student-teacher ratio + income + english + lunch + calworks + expenditure + computer"
    )
    print(f"  Coefficient on ratio = {coef_stratio_ctrl:.3f}")
    print(f"  Standard error      = {se_stratio_ctrl:.3f}")
    print(f"  p-value             = {p_stratio_ctrl:.3g}")
    print()

    # Provide a small JSON summary (printed) that can be reused when drafting the conclusion.
    summary = {
        "correlation": {"r": float(r), "p_value": float(p)},
        "simple_regression": {
            "coef_stratio": float(coef_stratio_simple),
            "se_stratio": float(se_stratio_simple),
            "p_value_stratio": float(p_stratio_simple),
        },
        "controls_regression": {
            "coef_stratio": float(coef_stratio_ctrl),
            "se_stratio": float(se_stratio_ctrl),
            "p_value_stratio": float(p_stratio_ctrl),
        },
    }

    print("JSON summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


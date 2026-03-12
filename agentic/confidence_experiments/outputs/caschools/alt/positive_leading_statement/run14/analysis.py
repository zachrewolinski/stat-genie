import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    base = Path(__file__).resolve().parent
    df = pd.read_csv(base / "caschools.csv")

    # Construct student-teacher ratio and overall test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    # Drop any rows with missing data in variables used (should be none but safe)
    cols = [
        "testscr",
        "stratio",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
        "income",
        "english",
    ]
    df_model = df[cols].dropna().copy()

    # Simple bivariate regression: testscr ~ stratio
    model_simple = smf.ols("testscr ~ stratio", data=df_model).fit()
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    # Multiple regression with key controls
    formula_controls = (
        "testscr ~ stratio + calworks + lunch + computer + "
        "expenditure + income + english"
    )
    model_controls = smf.ols(formula_controls, data=df_model).fit()
    coef_ctrl = model_controls.params["stratio"]
    pval_ctrl = model_controls.pvalues["stratio"]
    r2_ctrl = model_controls.rsquared

    # Correlation
    corr = df_model["testscr"].corr(df_model["stratio"])

    # Basic descriptives
    desc = df_model[["testscr", "stratio"]].describe().to_dict()

    summary = {
        "coef_simple": float(coef_simple),
        "pval_simple": float(pval_simple),
        "r2_simple": float(r2_simple),
        "coef_ctrl": float(coef_ctrl),
        "pval_ctrl": float(pval_ctrl),
        "r2_ctrl": float(r2_ctrl),
        "corr": float(corr),
        "desc": desc,
    }

    # Store numerical results for downstream use
    with open(base / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

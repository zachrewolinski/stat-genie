import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Define key variables
    students = df["feature6"]
    teachers = df["feature7"]

    # Guard against division by zero
    stratio = students / teachers.replace(0, np.nan)

    read_score = df["feature14"]
    math_score = df["feature15"]
    testscr = (read_score + math_score) / 2.0

    # Drop observations with missing values in key variables
    mask = stratio.notna() & testscr.notna()
    df_model = pd.DataFrame(
        {
            "testscr": testscr[mask],
            "stratio": stratio[mask],
            "calworks": df.loc[mask, "feature8"],
            "lunch": df.loc[mask, "feature9"],
            "computer": df.loc[mask, "feature10"],
            "expn_stu": df.loc[mask, "feature11"],
            "avginc": df.loc[mask, "feature12"],
            "el_pct": df.loc[mask, "feature13"],
        }
    )

    # Simple (bivariate) regression: test score on student-teacher ratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()

    # Multiple regression controlling for key covariates
    X_controls = df_model[
        ["stratio", "calworks", "lunch", "computer", "expn_stu", "avginc", "el_pct"]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["testscr"], X_controls).fit()

    # Extract key statistics
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]
    r2_simple = model_simple.rsquared

    coef_ctrl = model_controls.params["stratio"]
    pval_ctrl = model_controls.pvalues["stratio"]
    r2_ctrl = model_controls.rsquared

    # Basic standardized effect size: change in testscr per 1 SD in ratio
    str_sd = df_model["stratio"].std(ddof=0)
    testscr_sd = df_model["testscr"].std(ddof=0)
    effect_per_sd = coef_ctrl * str_sd / testscr_sd if testscr_sd > 0 else np.nan

    summary = {
        "n": int(df_model.shape[0]),
        "coef_simple": float(coef_simple),
        "pval_simple": float(pval_simple),
        "r2_simple": float(r2_simple),
        "coef_controls": float(coef_ctrl),
        "pval_controls": float(pval_ctrl),
        "r2_controls": float(r2_ctrl),
        "stratio_sd": float(str_sd),
        "testscr_sd": float(testscr_sd),
        "standardized_effect_per_sd": float(effect_per_sd)
        if not np.isnan(effect_per_sd)
        else None,
    }

    # Save a machine-readable summary for manual interpretation later if needed
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

